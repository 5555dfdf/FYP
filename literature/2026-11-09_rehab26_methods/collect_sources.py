"""Collect the verified key sources; no network activity without --fetch.

Run with `conda activate fyp` and `python -X utf8 .../collect_sources.py`.
New downloads go into a separate timestamped directory, preserving reviewed files.
Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


TARGETS = {
    "rehab26_paper.html": "https://arxiv.org/html/2609.35726v2",
    "rehab26_paper.pdf": "https://arxiv.org/pdf/2609.35726v2",
    "rehab26_zenodo.json": "https://zenodo.org/api/records/23018128",
    "openalex_exact.json": "https://api.openalex.org/works?search=REHAB26-ViewAngles&per-page=10",
    "openalex_broad.json": "https://api.openalex.org/works?search=REHAB26&per-page=25",
    "citing_dataset.json": "https://api.openalex.org/works?filter=cites:W7215103942&per-page=50",
    "citing_paper.json": "https://api.openalex.org/works?filter=cites:W7214708068&per-page=50",
    "personalized_springer.html": "https://link.springer.com/chapter/10.1007/978-3-031-75823-2_7",
    "personalized_crossref.json": "https://api.crossref.org/works?query.title="
    + quote("Personalized similarity models for evaluating rehabilitation exercises from monocular videos")
    + "&rows=2",
    "fine_tuning_crossref.json": "https://api.crossref.org/works/10.1016/j.is.2025.102579",
    "fine_tuning_elsevier.xml": "https://api.elsevier.com/content/article/PII:S0306437925000638?httpAccept=text/xml",
    "mobile_openalex.json": "https://api.openalex.org/works/https://doi.org/10.1145/3805622.3810429",
}


def fetch_one(item: tuple[str, str], directory: Path) -> dict:
    name, url = item
    entry = {
        "file": name,
        "url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (academic source verification)"})
        with urlopen(request, timeout=30) as response:
            body = response.read()
            entry.update(
                status=response.status,
                final_url=response.url,
                content_type=response.headers.get("Content-Type"),
            )
        if name.endswith(".pdf") and not body.startswith(b"%PDF"):
            raise ValueError("PDF endpoint returned a non-PDF response")
        if name.endswith(".json"):
            json.loads(body)
        (directory / name).write_bytes(body)
        entry.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    except Exception as error:
        entry["error"] = f"{type(error).__name__}: {error}"
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download sources into a new directory")
    args = parser.parse_args()
    if not args.fetch:
        print(json.dumps(TARGETS, ensure_ascii=False, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path(__file__).resolve().parent / "sources" / f"recheck_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(lambda item: fetch_one(item, directory), TARGETS.items()))
    (directory / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for entry in entries:
        print(json.dumps(entry, ensure_ascii=False))
    print(f"Saved: {directory}")


if __name__ == "__main__":
    main()
