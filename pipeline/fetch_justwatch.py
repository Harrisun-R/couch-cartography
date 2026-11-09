#!/usr/bin/env python3
"""Stage 1b - where-to-watch from JustWatch, via simple-justwatch-python-api.

Runs after fetch.py, on the same SQLite cache. For each of the top N shows:

1. Match: search JustWatch by title and keep the result whose TMDB id equals
   ours (JustWatch returns TMDB ids, so there is no guessing on similar
   titles). Tries the English title, then the original title. A match is
   stored for good; a miss is retried after --retry-days.
2. Offers: one request per show returns offers for every country at once.
   Re-fetched when older than --max-age days, oldest first.

When JustWatch has nothing for a show, build_map.py falls back to TMDB's
where-to-watch data (also from JustWatch, already fetched by fetch.py).
To fix a wrong or missing match by hand, see pipeline/justwatch_overrides.json.

The library uses JustWatch's unofficial GraphQL API, which can change or
refuse requests without notice. This script therefore never fails the build:
after too many errors in a row it stops, and the map uses what it has.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import DEFAULT_DB, RateLimiter, open_db  # noqa: E402

try:
    import simplejustwatchapi as jw
except ImportError:  # keep the build alive; build_map falls back to TMDB
    jw = None

ROOT = Path(__file__).resolve().parent.parent
OVERRIDES = ROOT / "pipeline" / "justwatch_overrides.json"
# JustWatch monetization types -> our keys (stream, free, rent, buy)
OFFER_KEYS = {"FLATRATE": "s", "FREE": "f", "ADS": "f", "FAST": "f", "RENT": "r", "BUY": "b"}
# Every country JustWatch serves (all 139 accepted in one query).
COUNTRIES = set(
    "AD AE AG AL AO AR AT AU AZ BA BB BE BF BG BH BM BO BR BS BY BZ CA CD CH CI CL CM CO "
    "CR CU CV CY CZ DE DK DO DZ EC EE EG ES FI FJ FR GB GF GG GH GI GQ GR GT GY HK HN HR "
    "HU ID IE IL IN IQ IS IT JM JO JP KE KR KW LB LC LI LT LU LV LY MA MC MD ME MG MK ML "
    "MT MU MW MX MY MZ NE NG NI NL NO NZ OM PA PE PF PG PH PK PL PS PT PY QA RO RS RU SA "
    "SC SE SG SI SK SM SN SV TC TD TH TN TR TT TW TZ UA UG US UY VA VE XK YE ZA ZM ZW".split()
)
MAX_ERRORS_IN_A_ROW = 25


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def days_ago(n: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=n)).isoformat(timespec="seconds")


def ensure_table(con):
    con.execute(
        """CREATE TABLE IF NOT EXISTS jw (
            id INTEGER PRIMARY KEY,   -- TMDB id
            node TEXT,                -- JustWatch id, e.g. ts4
            url TEXT,                 -- JustWatch page path, e.g. /us/tv-show/breaking-bad
            status TEXT,              -- ok | nomatch | manual | off
            matched_at TEXT,
            offers_at TEXT,
            offers TEXT               -- {"IN": {"s": [8], "b": [2]}, ...}
        )"""
    )


class Client:
    """Rate-limited calls with retries. Counts consecutive failures."""

    def __init__(self, rate: float):
        self.limiter = RateLimiter(rate)
        self.errors_in_a_row = 0

    def call(self, fn, *args, **kw):
        for attempt in range(4):
            self.limiter.wait()
            try:
                out = fn(*args, **kw)
                self.errors_in_a_row = 0
                return out
            except jw.JustWatchApiError:
                break  # the query itself was rejected; retrying won't help
            except Exception:  # network errors, 429, 5xx
                time.sleep(2 ** attempt * 2)
        self.errors_in_a_row += 1
        raise RuntimeError("JustWatch request failed")


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "").casefold()
    return re.sub(r"[^\w]+", " ", re.sub(r"^(the|a|an) ", "", t)).strip()


def find_node(client: Client, show: dict):
    """Return (node, url) for this show's JustWatch entry, or None.

    JustWatch's stored TMDB id is sometimes wrong (e.g. Glee, Cheers), so a
    result also counts if its IMDb id matches, or failing that if its title
    matches exactly and it started within a year of ours. Best match wins.
    """
    want, imdb = str(show["id"]), show.get("imdb_id") or ""
    year = int((show.get("first_air_date") or "0")[:4] or 0)
    names = [show["name"]]
    if show.get("original_name") and show["original_name"] != show["name"]:
        names.append(show["original_name"])
    titles = {norm_title(n) for n in names}
    countries = ["US"]
    if show.get("origin_country") and show["origin_country"][0] not in countries:
        countries.append(show["origin_country"][0])
    best = None  # (rank, node, url): 0 = TMDB id, 1 = IMDb id, 2 = title + year
    failed = False
    for name in names:
        for cc in countries:
            try:
                results = client.call(jw.search, name, country=cc, count=10, object_types="SHOW")
            except RuntimeError:
                if client.errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                    raise
                failed = True
                continue
            for e in results:
                if e.tmdb_id == want:
                    return e.entry_id, e.url or ""
                if imdb and e.imdb_id == imdb:
                    rank = 1
                elif year and e.release_year and abs(e.release_year - year) <= 1 and norm_title(e.title) in titles:
                    rank = 2
                else:
                    continue
                if best is None or rank < best[0]:
                    best = (rank, e.entry_id, e.url or "")
            if best and best[0] == 1:
                return best[1], best[2]
    if best:
        return best[1], best[2]
    if failed:  # a search didn't run, so "not on JustWatch" isn't known yet
        raise RuntimeError("JustWatch search failed")
    return None


def get_offers(client: Client, node: str):
    """{cc: {s|f|r|b: [package ids]}} plus {package id: (name, icon url)}."""
    raw = client.call(jw.offers_for_countries, node, COUNTRIES, best_only=True)
    out, packages = {}, {}
    for cc, offers in raw.items():
        kinds: dict[str, list[int]] = {}
        for o in offers:
            k = OFFER_KEYS.get(o.monetization_type)
            if not k or not o.package:
                continue
            pid = o.package.package_id
            if pid not in kinds.setdefault(k, []):
                kinds[k].append(pid)
            packages[pid] = (o.package.name, o.package.icon)
        if kinds:
            out[cc.upper()] = kinds
    return out, packages


def load_overrides():
    """{"1396": "ts4"} pins a match; {"1234": null} turns JustWatch off for a show."""
    try:
        raw = json.loads(OVERRIDES.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Ignoring {OVERRIDES.name}: not valid JSON ({e})")
        return {}
    return {int(k): v for k, v in raw.items() if str(k).isdigit()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--top-n", type=int, default=20000)
    ap.add_argument("--max-age", type=float, default=6, help="re-fetch offers older than this many days")
    ap.add_argument("--retry-days", type=float, default=30, help="retry shows with no match after this many days")
    ap.add_argument("--minutes", type=float, default=150, help="stop starting new work after this long")
    ap.add_argument("--rate", type=float, default=4, help="requests per second")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if jw is None:
        print("simple-justwatch-python-api isn't installed; skipping (the map will use TMDB's data).")
        return
    con = open_db(args.db)
    ensure_table(con)
    deadline = time.monotonic() + args.minutes * 60

    top = [(i, json.loads(d)) for i, d in con.execute(
        "SELECT id, data FROM shows WHERE data IS NOT NULL ORDER BY popularity DESC LIMIT ?", (args.top_n,))]
    rows = {r[0]: r for r in con.execute("SELECT id, node, status, matched_at, offers_at FROM jw")}

    # --- hand-made overrides win
    pinned = load_overrides()
    for sid, node in pinned.items():
        cur = rows.get(sid)
        if node is None:
            con.execute("INSERT OR REPLACE INTO jw (id, status, matched_at) VALUES (?, 'off', ?)", (sid, now()))
        elif not cur or cur[1] != node or cur[2] != "manual":
            con.execute("INSERT OR REPLACE INTO jw (id, node, url, status, matched_at) VALUES (?,?,?,'manual',?)",
                        (sid, node, "", now()))
    # an override removed from the file goes back to automatic matching
    for (sid,) in con.execute("SELECT id FROM jw WHERE status IN ('manual','off')").fetchall():
        if sid not in pinned:
            con.execute("DELETE FROM jw WHERE id=?", (sid,))
    con.commit()
    rows = {r[0]: r for r in con.execute("SELECT id, node, status, matched_at, offers_at FROM jw")}

    retry_before = days_ago(args.retry_days)
    to_match = [sh for sid, sh in top if sid not in rows
                or (rows[sid][2] == "nomatch" and (rows[sid][3] or "") < retry_before)]
    stale_before = days_ago(args.max_age)
    client = Client(args.rate)
    stats = {"matched": 0, "nomatch": 0, "offers": 0, "failed": 0}

    def run(jobs, work, label):
        """Run work(job) on a pool until done, out of time, or JustWatch keeps failing."""
        if not jobs:
            return True
        print(f"{label}: {len(jobs)} shows")
        it = iter(jobs)
        done = 0
        with ThreadPoolExecutor(args.workers) as pool:
            running = set()
            while True:
                while len(running) < args.workers * 2 and time.monotonic() < deadline \
                        and client.errors_in_a_row < MAX_ERRORS_IN_A_ROW:
                    job = next(it, None)
                    if job is None:
                        break
                    running.add(pool.submit(work, job))
                if not running:
                    break
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    res = f.result()
                    if res:
                        con.execute(*res)
                    done += 1
                    if done % 250 == 0:
                        con.commit()
                        print(f"  {done}/{len(jobs)}  {stats}", flush=True)
        con.commit()
        if client.errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
            print(f"JustWatch failed {MAX_ERRORS_IN_A_ROW} times in a row; stopping. "
                  "The map will use saved JustWatch data and TMDB's data.")
            return False
        if time.monotonic() >= deadline:
            print(f"Time budget used up after {done}/{len(jobs)}; the rest continue next run.")
            return False
        return True

    def match(sh):
        try:
            hit = find_node(client, sh)
        except RuntimeError:
            stats["failed"] += 1
            return None
        if hit:
            stats["matched"] += 1
            return ("INSERT OR REPLACE INTO jw (id, node, url, status, matched_at) VALUES (?,?,?,'ok',?)",
                    (sh["id"], hit[0], hit[1], now()))
        stats["nomatch"] += 1
        return ("INSERT OR REPLACE INTO jw (id, status, matched_at) VALUES (?, 'nomatch', ?)", (sh["id"], now()))

    packages_seen = {}

    def offers(job):
        sid, node = job
        try:
            data, packages = get_offers(client, node)
        except RuntimeError:
            stats["failed"] += 1
            return None
        stats["offers"] += 1
        packages_seen.update(packages)
        return ("UPDATE jw SET offers=?, offers_at=? WHERE id=?",
                (json.dumps(data, separators=(",", ":")), now(), sid))

    ok = run(to_match, match, "Matching shows to JustWatch")
    if ok:
        top_ids = [sid for sid, _ in top]
        cur = {r[0]: r for r in con.execute(
            "SELECT id, node, offers_at FROM jw WHERE status IN ('ok','manual') AND node IS NOT NULL")}
        stale = sorted(((sid, cur[sid][1]) for sid in top_ids
                        if sid in cur and (cur[sid][2] or "") < stale_before),
                       key=lambda j: cur[j[0]][2] or "")
        run(stale, offers, "Fetching JustWatch offers")

    # names and icons for services TMDB hasn't told us about (ids are shared with TMDB)
    known = {r[0] for r in con.execute("SELECT id FROM providers")}
    con.executemany("INSERT INTO providers (id, name, logo, priority) VALUES (?,?,?,999)",
                    [(pid, name, icon) for pid, (name, icon) in packages_seen.items() if pid not in known])
    con.commit()

    counts = dict(con.execute("SELECT status, COUNT(*) FROM jw GROUP BY status").fetchall())
    print(f"Done. This run: {stats}. Cache: {counts}")


if __name__ == "__main__":
    main()
