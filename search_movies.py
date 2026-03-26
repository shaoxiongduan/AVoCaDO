#!/usr/bin/env python3
"""Search for movie torrents given a list of movie names.

Searches the YTS API (movie-specific, free, no auth) to find magnet links
for each movie. Outputs a magnets.txt file ready to feed into download_movies.py.

For research purposes only.

Usage:
    python search_movies.py movies.txt
    python search_movies.py movies.txt --output magnets.txt --quality 1080p
    python search_movies.py movies.txt --min-seeds 5
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

YTS_BASE = "https://yts.bz/api/v2"
YTS_TRACKERS = [
    "udp://open.demonii.com:1337/announce",
    "udp://tracker.openbittorrent.com:80",
    "udp://tracker.coppersurfer.tk:6969",
    "udp://glotorrents.pw:6969/announce",
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://torrent.gresille.org:80/announce",
    "udp://p4p.arenabg.com:1337",
    "udp://tracker.leechers-paradise.org:6969",
]


@dataclass
class TorrentResult:
    movie_name: str
    torrent_title: str
    magnet: str
    quality: str
    size: str
    seeds: int


# ---------------------------------------------------------------------------
# YTS API
# ---------------------------------------------------------------------------

def _make_magnet(info_hash: str, title: str) -> str:
    """Build a magnet URI from a YTS hash."""
    trackers = "&".join(f"tr={quote(t)}" for t in YTS_TRACKERS)
    return f"magnet:?xt=urn:btih:{info_hash}&dn={quote(title)}&{trackers}"


def search_yts(query: str, quality: str | None = None) -> list[TorrentResult]:
    """Search the YTS API for a movie."""
    params = {"query_term": query, "limit": 10, "sort_by": "seeds", "order_by": "desc"}
    if quality:
        params["quality"] = quality

    try:
        resp = requests.get(f"{YTS_BASE}/list_movies.json", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {})
    except Exception as e:
        logger.warning(f"  YTS request failed: {e}")
        return []

    movies = data.get("movies") or []
    results = []
    for movie in movies:
        for torrent in movie.get("torrents", []):
            if quality and torrent.get("quality") != quality:
                continue
            info_hash = torrent.get("hash", "")
            title = f"{movie['title']} ({movie.get('year', '')})"
            results.append(TorrentResult(
                movie_name=query,
                torrent_title=title,
                magnet=_make_magnet(info_hash, title),
                quality=torrent.get("quality", "?"),
                size=torrent.get("size", "?"),
                seeds=int(torrent.get("seeds", 0)),
            ))
    return results


# ---------------------------------------------------------------------------
# Ranking & selection
# ---------------------------------------------------------------------------

def pick_best(results: list[TorrentResult], quality: str | None = None) -> TorrentResult | None:
    """Pick the best torrent from a list of results."""
    if not results:
        return None

    quality_rank = {"2160p": 4, "4k": 4, "1080p": 3, "720p": 2, "480p": 1, "?": 0}

    if quality:
        preferred = [r for r in results if r.quality == quality]
        if preferred:
            results = preferred

    # Sort by: preferred quality, then seeds
    results.sort(key=lambda r: (quality_rank.get(r.quality, 0), r.seeds), reverse=True)
    return results[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search for movie torrents and output magnet links."
    )
    parser.add_argument("movies_file", help="Text file with one movie name per line")
    parser.add_argument("-o", "--output", default="magnets.txt",
                        help="Output file for magnet links (default: magnets.txt)")
    parser.add_argument("--quality", default=None, choices=["720p", "1080p", "2160p"],
                        help="Preferred quality (default: best available)")
    parser.add_argument("--min-seeds", type=int, default=1,
                        help="Minimum seeders to consider (default: 1)")
    parser.add_argument("--results-json", default=None,
                        help="Optional: save full results to JSON for inspection")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay in seconds between searches (default: 1.0)")
    args = parser.parse_args()

    with open(args.movies_file) as f:
        movies = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    logger.info(f"Searching for {len(movies)} movies (quality={args.quality})")

    all_results = {}
    magnets = []
    not_found = []

    for idx, movie in enumerate(movies, 1):
        logger.info(f"[{idx}/{len(movies)}] Searching: {movie}")

        results = search_yts(movie, quality=args.quality)
        logger.info(f"  YTS: {len(results)} results")

        # Filter by minimum seeds
        results = [r for r in results if r.seeds >= args.min_seeds]

        best = pick_best(results, quality=args.quality)
        if best:
            logger.info(f"  -> {best.torrent_title} [{best.quality}] "
                        f"({best.size}, {best.seeds} seeds)")
            magnets.append((movie, best.magnet))
        else:
            logger.warning("  -> No results found")
            not_found.append(movie)

        all_results[movie] = [asdict(r) for r in results]

        if idx < len(movies):
            time.sleep(args.delay)

    # Write magnets file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for movie, magnet in magnets:
            f.write(f"# {movie}\n")
            f.write(f"{magnet}\n")

    logger.info(f"\nFound magnets for {len(magnets)}/{len(movies)} movies -> {output_path}")
    if not_found:
        logger.warning(f"Not found ({len(not_found)}): {', '.join(not_found)}")

    # Optionally save full results JSON
    if args.results_json:
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Full results saved to {args.results_json}")


if __name__ == "__main__":
    main()
