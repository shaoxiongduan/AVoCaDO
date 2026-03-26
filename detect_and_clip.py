#!/usr/bin/env python3
"""Detect scene transitions and generate 5-second clips from raw movies.

Continuously watches data/raw_movies/ for new movies, runs scene detection
and clip generation, and saves clips to data/clips/{movie_name}/. Fully
resumable — skips movies that already have clip_metadata.json.

Usage:
    python detect_and_clip.py
    python detect_and_clip.py --input data/raw_movies --output data/clips
    python detect_and_clip.py --clip-duration 5 --skip-start 300 --threshold 27
    python detect_and_clip.py --workers 4 --poll-interval 60
    python detect_and_clip.py --poll-interval 0   # exit when done (no polling)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    HashDetector,
    HistogramDetector,
    SceneManager,
    ThresholdDetector,
    open_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

def detect_transitions(
    video_path: Path,
    detector_type: str = "content",
    threshold: Optional[float] = None,
    min_scene_len: Optional[int] = None,
    downscale_factor: Optional[int] = None,
    frame_skip: int = 0,
) -> list[float]:
    """Detect scene transitions. Returns sorted list of timestamps in seconds."""
    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold
    if min_scene_len is not None:
        kwargs["min_scene_len"] = min_scene_len

    match detector_type:
        case "hash":
            detector = HashDetector(**kwargs)
        case "content":
            detector = ContentDetector(**kwargs)
        case "adaptive":
            if "threshold" in kwargs:
                kwargs["adaptive_threshold"] = kwargs.pop("threshold")
            detector = AdaptiveDetector(**kwargs)
        case "threshold":
            detector = ThresholdDetector(**kwargs)
        case "histogram":
            detector = HistogramDetector(**kwargs)
        case _:
            raise ValueError(f"Unknown detector type: {detector_type}")

    video = open_video(str(video_path), backend="opencv")
    scene_manager = SceneManager()

    if downscale_factor is not None:
        scene_manager.auto_downscale = False
        scene_manager.downscale = downscale_factor

    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video=video, show_progress=True, frame_skip=frame_skip)

    scenes = scene_manager.get_scene_list()
    transitions = [start.get_seconds() for i, (start, _end) in enumerate(scenes) if i > 0]
    return sorted(transitions)


# ---------------------------------------------------------------------------
# Video info via ffprobe
# ---------------------------------------------------------------------------

def get_video_info(video_path: Path) -> tuple[float, float]:
    """Get (duration_sec, fps) using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate:format=duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])
    num, den = info["streams"][0]["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return duration, fps


# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------

def generate_clips(
    video_path: Path,
    transitions: list[float],
    output_dir: Path,
    clip_duration: float = 5.0,
    max_internal_transitions: int = 1,
    skip_start: float = 300.0,
    skip_end: float = 300.0,
    num_frames: int = 121,
) -> list[dict]:
    """Generate clips at transition points. Returns list of clip metadata dicts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_duration, fps = get_video_info(video_path)
    video_stem = video_path.stem
    cutoff = video_duration - skip_end

    before = len(transitions)
    transitions = [t for t in transitions if t >= skip_start and t + clip_duration <= cutoff]
    skipped = before - len(transitions)
    if skipped:
        logger.info(f"  Filtered {skipped} transitions (before {skip_start:.0f}s intro / after {cutoff:.0f}s credits)")

    clips_metadata = []

    for idx, t_start in enumerate(transitions):
        t_end = t_start + clip_duration

        # Count other transitions within the clip window
        internal_count = sum(1 for t in transitions if t > t_start and t <= t_end)
        if internal_count > max_internal_transitions:
            continue

        clip_name = f"{video_stem}_clip_{idx:04d}_{t_start:.2f}s.mp4"
        clip_path = output_dir / clip_name

        # Skip clips that already exist (resume support within a movie)
        if clip_path.exists() and clip_path.stat().st_size > 0:
            meta = {
                "clip_path": str(clip_path),
                "source_video": str(video_path),
                "start_time": t_start,
                "end_time": t_end,
                "num_internal_transitions": internal_count,
                "num_frames": num_frames,
            }
            clips_metadata.append(meta)
            continue

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t_start),
            "-i", str(video_path),
            "-vf", (
                "fps=24,"
                "scale=1920:1088:force_original_aspect_ratio=decrease,"
                "pad=1920:1088:(ow-iw)/2:(oh-ih)/2"
            ),
            "-frames:v", str(num_frames),
            "-t", str(num_frames / 24),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k",
            str(clip_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"  ffmpeg failed for clip at {t_start:.2f}s: {e.stderr.decode()[:200]}")
            continue

        meta = {
            "clip_path": str(clip_path),
            "source_video": str(video_path),
            "start_time": t_start,
            "end_time": t_end,
            "num_internal_transitions": internal_count,
            "num_frames": num_frames,
        }
        clips_metadata.append(meta)

    return clips_metadata


# ---------------------------------------------------------------------------
# Per-movie worker (used by both single and multi-process modes)
# ---------------------------------------------------------------------------

def process_movie(
    video_path: Path,
    output_dir: Path,
    detector_type: str,
    threshold: Optional[float],
    min_scene_len: Optional[int],
    downscale_factor: Optional[int],
    frame_skip: int,
    clip_duration: float,
    max_internal_transitions: int,
    skip_start: float,
    skip_end: float,
    num_frames: int,
) -> dict:
    """Full detect-and-clip pipeline for a single movie. Returns summary dict."""
    movie_name = video_path.stem
    movie_out = output_dir / movie_name

    logger.info(f"Processing: {video_path.name}")

    # --- scene detection ---
    try:
        transitions = detect_transitions(
            video_path,
            detector_type=detector_type,
            threshold=threshold,
            min_scene_len=min_scene_len,
            downscale_factor=downscale_factor,
            frame_skip=frame_skip,
        )
    except Exception as e:
        logger.error(f"  Scene detection failed for {video_path.name}: {e}")
        return {"movie": movie_name, "error": str(e), "clips": 0}

    logger.info(f"  {len(transitions)} transitions detected")

    # --- clip generation ---
    clips = generate_clips(
        video_path,
        transitions,
        movie_out,
        clip_duration=clip_duration,
        max_internal_transitions=max_internal_transitions,
        skip_start=skip_start,
        skip_end=skip_end,
        num_frames=num_frames,
    )

    # Save per-movie metadata
    meta_path = movie_out / "clip_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(clips, f, indent=2)

    logger.info(f"  {len(clips)} clips saved to {movie_out}")
    return {"movie": movie_name, "transitions": len(transitions), "clips": len(clips)}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_pending(input_dir: Path, output_dir: Path) -> list[Path]:
    """Find movies in input_dir that don't yet have clip_metadata.json in output_dir."""
    if not input_dir.is_dir():
        return []
    all_videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    pending = []
    for v in all_videos:
        meta_file = output_dir / v.stem / "clip_metadata.json"
        if not meta_file.exists():
            pending.append(v)
    return pending


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect scene transitions and generate 5s clips from raw movies."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw_movies"),
                        help="Directory of movies (default: data/raw_movies)")
    parser.add_argument("--output", type=Path, default=Path("data/clips"),
                        help="Output root directory for clips (default: data/clips)")

    # Scene detection options
    parser.add_argument("--detector", type=str, default="hash",
                        choices=["hash", "content", "adaptive", "threshold", "histogram"],
                        help="Scene detector type (default: hash)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Detection threshold (default: detector's own default)")
    parser.add_argument("--min-scene-len", type=int, default=None,
                        help="Minimum scene length in frames")
    parser.add_argument("--downscale", type=int, default=None,
                        help="Downscale factor for detection speed")
    parser.add_argument("--frame-skip", type=int, default=0,
                        help="Frames to skip during detection (default: 0)")

    # Clip generation options
    parser.add_argument("--clip-duration", type=float, default=5.0,
                        help="Clip duration in seconds (default: 5.0)")
    parser.add_argument("--max-internal-transitions", type=int, default=1,
                        help="Max transitions allowed within a clip (default: 1)")
    parser.add_argument("--skip-start", type=float, default=300.0,
                        help="Skip transitions before this many seconds (default: 300)")
    parser.add_argument("--skip-end", type=float, default=300.0,
                        help="Skip transitions in the last N seconds / credits (default: 300)")
    parser.add_argument("--num-frames", type=int, default=121,
                        help="Exact frame count per clip (default: 121 = 5s@~24fps)")

    # Execution options
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between re-scans for new movies (0 to exit when done)")

    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common kwargs for process_movie
    kwargs = dict(
        output_dir=output_dir,
        detector_type=args.detector,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
        downscale_factor=args.downscale,
        frame_skip=args.frame_skip,
        clip_duration=args.clip_duration,
        max_internal_transitions=args.max_internal_transitions,
        skip_start=args.skip_start,
        skip_end=args.skip_end,
        num_frames=args.num_frames,
    )

    cumulative_results = []

    while True:
        videos = scan_pending(input_dir, output_dir)

        if not videos:
            if args.poll_interval <= 0:
                logger.info("All done. No pending movies.")
                break
            logger.info(f"No pending movies. Waiting {args.poll_interval}s...")
            time.sleep(args.poll_interval)
            continue

        logger.info(f"Found {len(videos)} new movies to process.")

        results = []

        if args.workers <= 1:
            for i, video in enumerate(videos, 1):
                logger.info(f"\n[{i}/{len(videos)}] ========================")
                result = process_movie(video_path=video, **kwargs)
                results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_movie, video_path=v, **kwargs): v
                    for v in videos
                }
                for future in as_completed(futures):
                    video = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Worker failed for {video.name}: {e}")
                        results.append({"movie": video.stem, "error": str(e), "clips": 0})

        # Batch summary
        total_clips = sum(r.get("clips", 0) for r in results)
        errors = [r for r in results if "error" in r]
        logger.info(f"\n--- Batch summary: {len(results)} movies, {total_clips} clips ---")
        if errors:
            for e in errors:
                logger.warning(f"  Error: {e['movie']}: {e['error']}")

        cumulative_results.extend(results)

        # Save cumulative summary (append-safe)
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(cumulative_results, f, indent=2)

        if args.poll_interval <= 0:
            break

        logger.info(f"Re-scanning in {args.poll_interval}s...")
        time.sleep(args.poll_interval)

    # Final summary
    total = sum(r.get("clips", 0) for r in cumulative_results)
    logger.info(f"\n========== FINAL SUMMARY ==========")
    logger.info(f"Total movies processed: {len(cumulative_results)}")
    logger.info(f"Total clips generated: {total}")


if __name__ == "__main__":
    main()
