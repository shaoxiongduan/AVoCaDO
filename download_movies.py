#!/usr/bin/env python3
"""Download movies from magnet links saved by the search stage.

Usage:
    python download_movies.py magnets.txt --output data/raw_movies
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.download.downloader import download_videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download movies from magnet links")
    parser.add_argument("magnets_file", help="File with one magnet URI per line")
    parser.add_argument("-o", "--output", default="data/raw_movies",
                        help="Output directory (default: data/raw_movies)")
    parser.add_argument("--format", default="mp4", help="Output format (default: mp4)")
    parser.add_argument("--max-resolution", default="1080", help="Max resolution (default: 1080)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Max seconds per torrent, 0=unlimited (default: 0)")
    args = parser.parse_args()

    with open(args.magnets_file) as f:
        magnets = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(magnets)} magnet links from {args.magnets_file}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = download_videos(
        sources=magnets,
        output_dir=output_dir,
        format=args.format,
        max_resolution=args.max_resolution,
        torrent_timeout=args.timeout,
    )
    logger.info(f"Download complete: {len(videos)} videos in {output_dir}")


if __name__ == "__main__":
    main()
