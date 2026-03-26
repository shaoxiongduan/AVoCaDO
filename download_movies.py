#!/usr/bin/env python3
"""Download movies from magnet links via BitTorrent using libtorrent.

Reads one magnet URI per line from a text file, downloads each torrent,
and keeps only the largest video file (the movie) in the output directory.

Usage:
    python download_movies.py magnets.txt --output data/raw_movies
    python download_movies.py magnets.txt --output data/raw_movies --timeout 3600
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

import libtorrent as lt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts", ".m4v"}


def download_torrent(
    session: lt.session,
    magnet: str,
    download_dir: Path,
    timeout: int = 0,
) -> list[Path]:
    """Download a single magnet link. Returns list of downloaded file paths."""
    params = lt.parse_magnet_uri(magnet)
    params.save_path = str(download_dir)
    handle = session.add_torrent(params)

    logger.info(f"  Fetching metadata for: {magnet[:80]}...")
    t0 = time.time()
    while not handle.status().has_metadata:
        time.sleep(1)
        if timeout and (time.time() - t0) > timeout:
            session.remove_torrent(handle)
            logger.warning("  Timed out waiting for metadata")
            return []

    info = handle.torrent_file()
    logger.info(f"  Downloading: {info.name()} ({info.total_size() / 1e9:.2f} GB)")

    while not handle.status().is_seeding:
        s = handle.status()
        elapsed = time.time() - t0
        logger.info(
            f"  {s.progress * 100:.1f}% | "
            f"down: {s.download_rate / 1e6:.1f} MB/s | "
            f"peers: {s.num_peers} | "
            f"elapsed: {elapsed:.0f}s"
        )
        if timeout and elapsed > timeout:
            logger.warning(f"  Timed out after {timeout}s ({s.progress * 100:.1f}% done)")
            break
        time.sleep(5)

    # Collect downloaded files
    files = []
    for i in range(info.num_files()):
        fp = download_dir / info.files().file_path(i)
        if fp.exists():
            files.append(fp)

    session.remove_torrent(handle)
    return files


def pick_video(files: list[Path]) -> Path | None:
    """Return the largest video file from a list of paths."""
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        return None
    return max(videos, key=lambda f: f.stat().st_size)


def download_all(
    magnets: list[str],
    output_dir: Path,
    timeout: int = 0,
) -> list[Path]:
    """Download all magnet links and collect the main video from each."""
    session = lt.session()
    session.listen_on(6881, 6891)

    # Basic settings for faster downloads
    settings = session.get_settings()
    settings["alert_mask"] = lt.alert.category_t.error_notification
    session.apply_settings(settings)

    staging_dir = output_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for idx, magnet in enumerate(magnets, 1):
        logger.info(f"[{idx}/{len(magnets)}] Starting download")
        try:
            files = download_torrent(session, magnet, staging_dir, timeout=timeout)
        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

        video = pick_video(files)
        if video is None:
            logger.warning(f"  No video file found in torrent")
            continue

        dest = output_dir / video.name
        if dest.exists():
            logger.info(f"  Already exists, skipping: {dest.name}")
        else:
            shutil.move(str(video), str(dest))
            logger.info(f"  Saved: {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")
        downloaded.append(dest)

    # Clean up staging directory
    shutil.rmtree(staging_dir, ignore_errors=True)
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download movies from magnet links")
    parser.add_argument("magnets_file", help="File with one magnet URI per line")
    parser.add_argument("-o", "--output", default="data/raw_movies",
                        help="Output directory (default: data/raw_movies)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Max seconds per torrent, 0=unlimited (default: 0)")
    args = parser.parse_args()

    with open(args.magnets_file) as f:
        magnets = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    logger.info(f"Loaded {len(magnets)} magnet links from {args.magnets_file}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = download_all(magnets, output_dir, timeout=args.timeout)
    logger.info(f"Download complete: {len(videos)}/{len(magnets)} videos saved to {output_dir}")


if __name__ == "__main__":
    main()
