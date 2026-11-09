#!/usr/bin/env python3
"""Stage 1 - fetch TV shows from TMDB into a local SQLite cache.

First run (empty cache): downloads TMDB's daily TV-series ID export, picks the
top N shows by popularity and fetches full details for each.

Later runs: refreshes popularity for every cached show from the export, fetches
any new entrants to the top N, and re-fetches every top-N show older than
--max-age days. The full weekly refresh keeps where-to-watch data current,
which TMDB's change feed doesn't reliably report.

Every record is committed as it arrives, so the script can be stopped and
restarted at any time without losing progress.

Auth: set TMDB_API_KEY to either a v3 API key or a v4 read access token.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://api.themoviedb.org/3"
EXPORT_URL = "https://files.tmdb.org/p/exports/tv_series_ids_{d:%m_%d_%Y}.json.gz"
APPEND = "keywords,aggregate_credits,content_ratings,external_ids,watch/providers"
# TMDB/JustWatch offer types -> our short keys (stream, free, rent, buy)
OFFER_KEYS = {"flatrate": "s", "free": "f", "ads": "f", "rent": "r", "buy": "b"}
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "cache" / "tmdb.sqlite"


# --------------------------------------------------------------------------- db
def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS shows (
            id INTEGER PRIMARY KEY,
            popularity REAL,
            fetched_at TEXT,
            data TEXT
        );
        CREATE TABLE IF NOT EXISTS missing (id INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS layout (id INTEGER PRIMARY KEY, x REAL, y REAL);
        CREATE TABLE IF NOT EXISTS providers (
            id INTEGER PRIMARY KEY, name TEXT, logo TEXT, priority INTEGER
        );
        """
    )
    return con


def meta_get(con, k, default=None):
    row = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return row[0] if row else default


def meta_set(con, k, v):
    con.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (k, str(v)))


# ----------------------------------------------------------------------- http
class RateLimiter:
    """Simple shared limiter: at most `rate` requests per second across threads."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_t = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        delay = t - now
        if delay > 0:
            time.sleep(delay)


class Tmdb:
    def __init__(self, key: str, rate: float):
        self.key = key.strip()
        self.bearer = self.key.startswith("eyJ")  # v4 tokens are JWTs
        self.limiter = RateLimiter(rate)
        self.local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self.local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers["Accept"] = "application/json"
            if self.bearer:
                s.headers["Authorization"] = f"Bearer {self.key}"
            self.local.s = s
        return s

    def get(self, path: str, **params):
        if not self.bearer:
            params["api_key"] = self.key
        for attempt in range(6):
            self.limiter.wait()
            try:
                r = self._session().get(f"{API}{path}", params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 401:
                sys.exit("TMDB rejected the API key (401). Check TMDB_API_KEY.")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            r.raise_for_status()
        raise RuntimeError(f"Gave up on {path} after retries")


# --------------------------------------------------------------------- export
def download_export() -> list[dict]:
    """Return the most recent daily export (tries today, then earlier days)."""
    today = dt.datetime.now(dt.timezone.utc).date()
    for back in range(4):
        url = EXPORT_URL.format(d=today - dt.timedelta(days=back))
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            print(f"Using export {url}")
            lines = gzip.decompress(r.content).decode("utf-8").splitlines()
            return [json.loads(l) for l in lines if l.strip()]
    sys.exit("Could not download a TMDB daily export from the last 4 days.")


# ----------------------------------------------------------------------- trim
def trim_providers(raw: dict):
    """Per-country offers as {cc: {s|f|r|b: [provider ids]}}, plus provider info."""
    offers, info = {}, {}
    for cc, entry in (raw.get("watch/providers", {}).get("results") or {}).items():
        out = {}
        for kind, key in OFFER_KEYS.items():
            for p in entry.get(kind) or []:
                pid = p["provider_id"]
                info[pid] = (p.get("provider_name", ""), p.get("logo_path", ""), p.get("display_priority", 999))
                if pid not in out.setdefault(key, []):
                    out[key].append(pid)
        if out:
            offers[cc] = out
    return offers, info


def trim(raw: dict) -> dict:
    """Keep only what the map needs; aggregate_credits can be enormous."""
    cast = sorted(
        raw.get("aggregate_credits", {}).get("cast", []),
        key=lambda c: (-(c.get("total_episode_count") or 0), c.get("order", 999)),
    )[:12]
    ratings = raw.get("content_ratings", {}).get("results", [])
    rating = next((r["rating"] for r in ratings if r.get("iso_3166_1") == "US"), None)
    if rating is None and ratings:
        rating = ratings[0].get("rating")
    return {
        "id": raw["id"],
        "name": raw.get("name") or raw.get("original_name") or "",
        "original_name": raw.get("original_name") or "",
        "overview": raw.get("overview") or "",
        "poster_path": raw.get("poster_path"),
        "first_air_date": raw.get("first_air_date") or "",
        "last_air_date": raw.get("last_air_date") or "",
        "genres": [g["name"] for g in raw.get("genres", [])],
        "networks": [n["name"] for n in raw.get("networks", [])],
        "origin_country": raw.get("origin_country") or [],
        "original_language": raw.get("original_language") or "",
        "number_of_seasons": raw.get("number_of_seasons") or 0,
        "number_of_episodes": raw.get("number_of_episodes") or 0,
        "episode_run_time": raw.get("episode_run_time") or [],
        "status": raw.get("status") or "",
        "type": raw.get("type") or "",
        "vote_average": raw.get("vote_average") or 0,
        "vote_count": raw.get("vote_count") or 0,
        "popularity": raw.get("popularity") or 0,
        "adult": bool(raw.get("adult")),
        "created_by": [c["name"] for c in raw.get("created_by", [])],
        "keywords": [k["name"] for k in raw.get("keywords", {}).get("results", [])],
        "cast": [c["name"] for c in cast],
        "content_rating": rating,
        "last_runtime": (raw.get("last_episode_to_air") or {}).get("runtime") or 0,
        "imdb_id": (raw.get("external_ids") or {}).get("imdb_id") or "",
        "tvdb_id": (raw.get("external_ids") or {}).get("tvdb_id") or 0,
        "providers": trim_providers(raw)[0],
    }


# ---------------------------------------------------------------------- fetch
def fetch_many(con, tmdb: Tmdb, ids: list[int], workers: int, label: str):
    if not ids:
        print(f"{label}: nothing to fetch")
        return
    print(f"{label}: fetching {len(ids)} shows")
    done = 0
    started = time.monotonic()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        with ThreadPoolExecutor(workers) as pool:
            futs = {pool.submit(tmdb.get, f"/tv/{i}", append_to_response=APPEND): i for i in ids}
            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    raw = fut.result()
                except Exception as e:  # noqa: BLE001 - log and move on
                    print(f"  ! {sid}: {e}")
                    continue
                if raw is None:
                    con.execute("INSERT OR IGNORE INTO missing (id) VALUES (?)", (sid,))
                    con.execute("DELETE FROM shows WHERE id=?", (sid,))
                else:
                    rec = trim(raw)
                    con.executemany(
                        "INSERT OR REPLACE INTO providers (id, name, logo, priority) VALUES (?,?,?,?)",
                        [(pid, *v) for pid, v in trim_providers(raw)[1].items()],
                    )
                    con.execute(
                        "INSERT OR REPLACE INTO shows (id, popularity, fetched_at, data) VALUES (?,?,?,?)",
                        (sid, rec["popularity"], now, json.dumps(rec, ensure_ascii=False)),
                    )
                done += 1
                if done % 250 == 0:
                    con.commit()
                    rate = done / (time.monotonic() - started)
                    print(f"  {done}/{len(ids)}  ({rate:.1f}/s)")
    finally:
        con.commit()
    print(f"{label}: done ({done})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--top-n", type=int, default=int(os.environ.get("TOP_N", 20000)),
                    help="how many shows (by popularity) to keep on the map")
    ap.add_argument("--rate", type=float, default=35.0, help="max requests per second")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-age", type=float, default=6.0,
                    help="re-fetch top-N shows fetched more than this many days ago")
    args = ap.parse_args()

    key = os.environ.get("TMDB_API_KEY")
    if not key:
        sys.exit("Set TMDB_API_KEY (v3 key or v4 read access token).")
    tmdb = Tmdb(key, args.rate)
    con = open_db(args.db)
    run_started = dt.datetime.now(dt.timezone.utc)

    export = download_export()
    pop = {row["id"]: row.get("popularity", 0) for row in export if not row.get("adult")}

    # 1. refresh popularity of everything we already have
    con.executemany("UPDATE shows SET popularity=? WHERE id=?", [(p, i) for i, p in pop.items()])
    con.commit()

    # 2. fetch top-N entrants we don't have yet (fetch a margin; some get filtered later)
    want = int(args.top_n * 1.15)
    have = {r[0] for r in con.execute("SELECT id FROM shows")}
    missing = {r[0] for r in con.execute("SELECT id FROM missing")}
    top = sorted(pop, key=pop.get, reverse=True)[:want]
    fetch_many(con, tmdb, [i for i in top if i not in have and i not in missing], args.workers, "New shows")

    # 3. weekly refresh: re-fetch every top-N show older than --max-age days
    cutoff = (run_started - dt.timedelta(days=args.max_age)).isoformat(timespec="seconds")
    stale = {r[0] for r in con.execute("SELECT id FROM shows WHERE fetched_at < ?", (cutoff,))}
    fetch_many(con, tmdb, [i for i in top if i in stale], args.workers, "Refresh")

    meta_set(con, "last_run", run_started.isoformat(timespec="seconds"))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM shows").fetchone()[0]
    print(f"Cache now holds {n} shows -> {args.db}")


if __name__ == "__main__":
    main()
