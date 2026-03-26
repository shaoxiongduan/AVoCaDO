"""
Batch-caption all video clips under a clips directory using AVoCaDO.

Continuously watches data/clips/ for new un-captioned clips, mirrors the
folder structure into data/captions/, saving one JSON per clip. Loads one
model per GPU and is fully resumable (skips clips that already have a
caption JSON). Re-scans for new clips after each batch.

Usage:
    # Single GPU:
    python caption_clip.py --gpus 0

    # Multi-GPU:
    python caption_clip.py --gpus 0,1,2,3

    # Run once (no polling):
    python caption_clip.py --gpus 0 --poll-interval 0
"""

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

# --- Constants ---
VIDEO_MAX_PIXELS = 401408       # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400   # 512*28*28*50
USE_AUDIO_IN_VIDEO = True

PROMPT_LIST = [
    "Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.",
    "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.",
    "Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.",
    "Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.",
    "Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.",
    "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.",
    "Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized.",
]


def get_video_metadata(clip_path: Path) -> dict:
    """Extract video metadata using ffprobe."""
    meta = {}
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(clip_path),
            ],
            capture_output=True, text=True,
        )
        probe = json.loads(result.stdout)
        fmt = probe.get("format", {})
        meta["duration_sec"] = float(fmt.get("duration", 0))
        meta["file_size_bytes"] = int(fmt.get("size", 0))
        for stream in probe.get("streams", []):
            if stream["codec_type"] == "video":
                meta["width"] = stream.get("width")
                meta["height"] = stream.get("height")
                meta["fps"] = stream.get("r_frame_rate")
                meta["video_codec"] = stream.get("codec_name")
                meta["num_frames"] = int(stream.get("nb_frames", 0))
            elif stream["codec_type"] == "audio":
                meta["audio_codec"] = stream.get("codec_name")
                meta["sample_rate"] = stream.get("sample_rate")
                meta["audio_channels"] = stream.get("channels")
    except Exception as e:
        meta["ffprobe_error"] = str(e)
    return meta


def load_clip_metadata(movie_dir: Path) -> dict:
    """Load clip_metadata.json and return a dict keyed by clip filename."""
    meta_file = movie_dir / "clip_metadata.json"
    if not meta_file.exists():
        return {}
    with open(meta_file) as f:
        entries = json.load(f)
    return {Path(e["clip_path"]).name: e for e in entries}


def scan_pending(clips_dir: Path, captions_dir: Path) -> List[tuple]:
    """Scan clips_dir for un-captioned clips. Returns (movie_dir, clip_path) pairs."""
    if not clips_dir.is_dir():
        return []
    todo = []
    for movie_dir in sorted(clips_dir.iterdir()):
        if not movie_dir.is_dir():
            continue
        for clip in sorted(movie_dir.glob("*.mp4")):
            out_json = captions_dir / movie_dir.name / f"{clip.stem}.json"
            if not out_json.exists():
                todo.append((movie_dir, clip))
    return todo


def _build_conversation(file_path: str, prompt: str) -> list:
    """Build a single conversation dict for one clip."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": file_path, "max_pixels": VIDEO_MAX_PIXELS},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _save_caption(clip_path, caption, prompt, model_path, captions_dir, meta_cache):
    """Write one caption JSON to disk."""
    movie_dir = clip_path.parent
    movie_name = movie_dir.name

    out_dir = captions_dir / movie_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{clip_path.stem}.json"

    video_meta = get_video_metadata(clip_path)

    if movie_name not in meta_cache:
        meta_cache[movie_name] = load_clip_metadata(movie_dir)
    source_meta = meta_cache[movie_name].get(clip_path.name)

    result = {
        "clip_name": clip_path.name,
        "caption": caption,
        "prompt_used": prompt,
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
        "source_clip_path": str(clip_path),
        "video_metadata": video_meta,
    }
    if source_meta:
        result["source_metadata"] = {
            "source_video": source_meta.get("source_video"),
            "start_time": source_meta.get("start_time"),
            "end_time": source_meta.get("end_time"),
            "num_internal_transitions": source_meta.get("num_internal_transitions"),
        }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)


def worker_process(
    gpu_id: int,
    assigned_clips: List[tuple],
    model_path: str,
    captions_dir: str,
    seed: int,
    batch_size: int = 1,
):
    """Per-GPU worker: loads model on one GPU, captions its share of clips."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VIDEO_MAX_PIXELS"] = str(VIDEO_TOTAL_PIXELS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from tqdm import tqdm
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    captions_dir = Path(captions_dir)

    print(f"[GPU {gpu_id}] Loading model from: {model_path}", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    print(f"[GPU {gpu_id}] Model loaded. Batch size: {batch_size}", flush=True)

    @torch.inference_mode()
    def generate_captions_batch(file_paths: List[str], prompts: List[str]) -> List[str]:
        conversations = [_build_conversation(fp, p) for fp, p in zip(file_paths, prompts)]
        texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        # Pad videos to same frame count (safety for any slight variation)
        if videos:
            max_frames = max(v.shape[0] for v in videos)
            padded = []
            for v in videos:
                if v.shape[0] < max_frames:
                    pad = torch.zeros((max_frames - v.shape[0], *v.shape[1:]), dtype=v.dtype)
                    padded.append(torch.cat([v, pad], dim=0))
                else:
                    padded.append(v)
            videos = padded

        inputs = processor(
            text=texts, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            do_sample=False,
            thinker_max_new_tokens=2048,
        )
        decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [d.split("\nassistant\n")[-1] for d in decoded]

    rng = random.Random(seed + gpu_id)
    meta_cache = {}

    # Filter out already-captioned clips (double-check at worker start)
    pending = []
    for movie_dir_str, clip_path_str in assigned_clips:
        clip_path = Path(clip_path_str)
        out_json = captions_dir / Path(movie_dir_str).name / f"{clip_path.stem}.json"
        if not out_json.exists():
            pending.append((movie_dir_str, clip_path_str))

    # Process in batches
    pbar = tqdm(total=len(pending), desc=f"[GPU {gpu_id}]", dynamic_ncols=True)

    for batch_start in range(0, len(pending), batch_size):
        chunk = pending[batch_start : batch_start + batch_size]
        clip_paths = [Path(c[1]) for c in chunk]
        prompts = [rng.choice(PROMPT_LIST) for _ in chunk]
        file_paths = [str(cp) for cp in clip_paths]

        try:
            captions = generate_captions_batch(file_paths, prompts)
            for clip_path, caption, prompt in zip(clip_paths, captions, prompts):
                _save_caption(clip_path, caption, prompt, model_path, captions_dir, meta_cache)
        except Exception as e:
            tqdm.write(f"[GPU {gpu_id}] Batch failed ({len(chunk)} clips), falling back to one-at-a-time: {e}")
            for clip_path, prompt in zip(clip_paths, prompts):
                try:
                    captions = generate_captions_batch([str(clip_path)], [prompt])
                    _save_caption(clip_path, captions[0], prompt, model_path, captions_dir, meta_cache)
                except Exception as e2:
                    tqdm.write(f"[GPU {gpu_id}] FAILED {clip_path.name}: {e2}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        pbar.update(len(chunk))

    pbar.close()
    print(f"[GPU {gpu_id}] Batch done.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Batch-caption video clips with AVoCaDO (multi-GPU)")
    parser.add_argument("--clips-dir", type=str, default="data/clips",
                        help="Path to clips directory (default: data/clips)")
    parser.add_argument("--captions-dir", type=str, default="data/captions",
                        help="Output captions directory (default: data/captions)")
    parser.add_argument("--model", type=str, default="AVoCaDO-Captioner/AVoCaDO",
                        help="Model path or HuggingFace ID")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed for prompt selection")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of clips per forward pass per GPU")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between re-scans for new clips (0 to exit when done)")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir).resolve()
    captions_dir = Path(args.captions_dir).resolve()
    captions_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]

    if not gpu_ids:
        print("No GPUs specified in --gpus", file=sys.stderr)
        sys.exit(1)

    print(f"Using {len(gpu_ids)} GPU(s): {gpu_ids}", flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    while True:
        todo = scan_pending(clips_dir, captions_dir)

        if not todo:
            if args.poll_interval <= 0:
                print("All done. No pending clips.")
                return
            print(f"No pending clips. Waiting {args.poll_interval}s for new clips...", flush=True)
            time.sleep(args.poll_interval)
            continue

        print(f"Found {len(todo)} pending clips. Distributing across {len(gpu_ids)} GPU(s).", flush=True)

        # Round-robin distribute clips across GPUs (serialize paths for spawn)
        buckets = [[] for _ in gpu_ids]
        for i, (movie_dir, clip_path) in enumerate(todo):
            buckets[i % len(gpu_ids)].append((str(movie_dir), str(clip_path)))

        procs = []
        for i, gpu_id in enumerate(gpu_ids):
            if not buckets[i]:
                continue
            p = ctx.Process(
                target=worker_process,
                args=(gpu_id, buckets[i], args.model, str(captions_dir), args.seed, args.batch_size),
            )
            p.start()
            procs.append(p)

        exit_codes = []
        for p in procs:
            p.join()
            exit_codes.append(p.exitcode)

        bad = [c for c in exit_codes if c != 0]
        if bad:
            print(f"WARNING: some workers exited non-zero: {exit_codes}", file=sys.stderr, flush=True)

        print("Batch done. Re-scanning for new clips...", flush=True)

        if args.poll_interval <= 0:
            break


if __name__ == "__main__":
    main()
