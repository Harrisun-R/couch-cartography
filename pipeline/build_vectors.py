#!/usr/bin/env python3
"""Stage 2 - turn cached shows into weighted feature vectors.

Picks the top N shows by current popularity, filters out thin records, and
builds one sparse vector per show from weighted blocks:

    genres, show type, keywords, networks, language, origin country,
    people (creators + main cast), and a few standardised numbers.

Each block is L2-normalised per show and then multiplied by its weight, so
a show with 40 keywords doesn't drown out its genres.

Outputs build/vectors.npz (sparse matrix) and build/shows.json (records in
the same row order).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "cache" / "tmdb.sqlite"
BUILD = ROOT / "build"

# block -> (weight, max vocabulary size, min shows needed to keep a term)
BLOCKS = {
    "genres":   (3.0, 64, 1),
    "type":     (1.5, 16, 1),
    "keywords": (2.2, 500, 5),
    "networks": (1.3, 200, 5),
    "language": (1.6, 50, 5),
    "country":  (0.8, 50, 5),
    "people":   (0.9, 400, 4),
}
NUMERIC_WEIGHT = 0.6
EXCLUDED_TYPES = {"News"}


def year(d: str) -> int:
    return int(d[:4]) if d and d[:4].isdigit() else 0


def terms(show: dict, block: str) -> list[str]:
    if block == "genres":
        return show["genres"]
    if block == "type":
        return [show["type"]] if show["type"] else []
    if block == "keywords":
        return show["keywords"]
    if block == "networks":
        return show["networks"]
    if block == "language":
        return [show["original_language"]] if show["original_language"] else []
    if block == "country":
        return show["origin_country"]
    if block == "people":
        return list(dict.fromkeys(show["created_by"] + show["cast"][:8]))
    raise KeyError(block)


def numeric(show: dict) -> list[float]:
    runtime = show["episode_run_time"]
    return [
        math.log1p(show["number_of_episodes"]),
        math.log1p(show["number_of_seasons"]),
        math.log1p(sum(runtime) / len(runtime)) if runtime else 0.0,
        float(year(show["first_air_date"]) or 2000),
        math.log1p(show["popularity"]),
    ]


def load_shows(db: Path, top_n: int, min_votes: int) -> list[dict]:
    con = sqlite3.connect(db)
    rows = con.execute("SELECT data, popularity FROM shows ORDER BY popularity DESC").fetchall()
    shows = []
    for data, pop in rows:
        s = json.loads(data)
        s["popularity"] = pop or s.get("popularity", 0)
        if s.get("adult") or s["type"] in EXCLUDED_TYPES:
            continue
        if not s["genres"] and not s["keywords"]:
            continue  # nothing to place it by
        if s["vote_count"] < min_votes and not s["overview"]:
            continue
        shows.append(s)
        if len(shows) >= top_n:
            break
    return shows


def build(shows: list[dict]):
    n = len(shows)
    blocks = []
    vocab_info = {}
    for block, (weight, max_vocab, min_df) in BLOCKS.items():
        df = Counter(t for s in shows for t in set(terms(s, block)))
        vocab = [t for t, c in df.most_common(max_vocab) if c >= min_df]
        index = {t: i for i, t in enumerate(vocab)}
        rows, cols = [], []
        for r, s in enumerate(shows):
            for t in set(terms(s, block)):
                if t in index:
                    rows.append(r)
                    cols.append(index[t])
        m = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, max(len(vocab), 1)))
        # idf so rare-but-present terms count more than ubiquitous ones
        idf = np.log((1 + n) / (1 + np.asarray(m.sum(axis=0)).ravel())) + 1
        m = m.multiply(idf.astype(np.float32)).tocsr()
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norms[norms == 0] = 1
        m = sparse.diags((weight / norms).astype(np.float32)) @ m
        blocks.append(m)
        vocab_info[block] = len(vocab)

    num = np.array([numeric(s) for s in shows], dtype=np.float32)
    num = (num - num.mean(0)) / (num.std(0) + 1e-6)
    num *= NUMERIC_WEIGHT / math.sqrt(num.shape[1])
    blocks.append(sparse.csr_matrix(num))
    vocab_info["numeric"] = num.shape[1]

    return sparse.hstack(blocks).tocsr().astype(np.float32), vocab_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--top-n", type=int, default=int(os.environ.get("TOP_N", 20000)))
    ap.add_argument("--min-votes", type=int, default=3)
    args = ap.parse_args()

    shows = load_shows(args.db, args.top_n, args.min_votes)
    if len(shows) < 50:
        raise SystemExit(f"Only {len(shows)} usable shows in the cache; run fetch.py first.")
    X, info = build(shows)
    BUILD.mkdir(exist_ok=True)
    sparse.save_npz(BUILD / "vectors.npz", X)
    (BUILD / "shows.json").write_text(json.dumps(shows, ensure_ascii=False))
    dims = ", ".join(f"{k} {v}" for k, v in info.items())
    print(f"{len(shows)} shows x {X.shape[1]} dims ({dims}) -> {BUILD}")


if __name__ == "__main__":
    main()
