#!/usr/bin/env python3
"""Stage 3 - project, cluster and export the map.

* TruncatedSVD (50-D) -> cosine kNN for "similar shows"
* UMAP 2-D for map positions. Positions from the previous run (kept in the
  SQLite `layout` table) seed the layout, then a Procrustes fit lines the new
  map up with the old one, so shows don't jump around between weekly runs.
* UMAP 10-D + HDBSCAN for clusters, named by their most distinctive terms.
* Writes site/data/points.json (everything needed to draw and filter),
  site/data/details/NN.json (loaded only when a show is opened) and
  site/data/providers/<CC>.json (where to watch, loaded per country).

Where to watch comes from TMDB's copy of JustWatch data (fetch.py). With
--justwatch it also uses data fetched directly from JustWatch
(fetch_justwatch.py, optional and off by default). Each show records which
source was used.
"""
from __future__ import annotations

import argparse
import colorsys
import datetime as dt
import json
import math
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "cache" / "tmdb.sqlite"
BUILD = ROOT / "build"
OUT = ROOT / "site" / "data"
DETAIL_CHUNKS = 40
SIMILAR = 20
NOTES = ROOT / "pipeline" / "cluster_notes.json"  # hand-written region names and blurbs
JW_FRESH_DAYS = 21  # older direct JustWatch data loses to TMDB's weekly copy

LANG = {
    "en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "cn": "Cantonese",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
    "tr": "Turkish", "hi": "Hindi", "ru": "Russian", "th": "Thai", "ar": "Arabic",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "nl": "Dutch", "pl": "Polish",
    "ta": "Tamil", "te": "Telugu", "tl": "Tagalog", "id": "Indonesian", "fi": "Finnish",
    "he": "Hebrew", "ml": "Malayalam", "uk": "Ukrainian", "vi": "Vietnamese", "fa": "Persian",
}


# ------------------------------------------------------------------ helpers
def year(d: str) -> int:
    return int(d[:4]) if d and d[:4].isdigit() else 0


# typical episode lengths, used when TMDB has no runtime for a show
RUNTIME_GUESS = [("Kids", 15), ("Animation", 24), ("Comedy", 25), ("Reality", 42),
                 ("Documentary", 48), ("Talk", 45), ("Soap", 40)]


def runtime(show) -> tuple[int, bool]:
    """Average episode length in minutes, and whether it is a guess."""
    rt = [r for r in show.get("episode_run_time") or [] if r]
    if rt:
        return round(sum(rt) / len(rt)), False
    if show.get("last_runtime"):
        return int(show["last_runtime"]), False
    for genre, mins in RUNTIME_GUESS:
        if genre in show["genres"]:
            return mins, True
    return 45, True


def total_hours(show) -> tuple[float, bool]:
    mins, guess = runtime(show)
    eps = show["number_of_episodes"] or 0
    return round(eps * mins / 60, 1), guess


def umap_fit(Z, n_components, init="spectral", min_dist=0.1, seed=None, gentle=False):
    import umap  # imported late: numba start-up is slow

    extra = {"learning_rate": 0.1, "n_epochs": 60} if gentle else {}
    return umap.UMAP(
        n_components=n_components, n_neighbors=15, min_dist=min_dist,
        metric="cosine", init=init, random_state=seed, low_memory=True, **extra,
    ).fit_transform(Z)


def procrustes_align(new, old):
    """Rotate/scale/shift `new` (all points) so its rows in `old`'s order match."""
    mu_n, mu_o = new.mean(0), old.mean(0)
    a, b = new - mu_n, old - mu_o
    u, s, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    scale = s.sum() / (a ** 2).sum()
    return r, scale, mu_n, mu_o


def hsl_hex(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def label_terms(show):
    out = [("g", g) for g in show["genres"]]
    out += [("k", k) for k in show["keywords"]]
    out += [("n", n) for n in show["networks"]]
    if show["original_language"] and show["original_language"] != "en":
        out.append(("l", show["original_language"]))
    return set(out)


def pretty(term):
    kind, value = term
    if kind == "l":
        return LANG.get(value, value.upper())
    if kind == "k":
        return value[:1].upper() + value[1:]
    return value


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--fresh", action="store_true", help="ignore the previous layout")
    ap.add_argument("--justwatch", action="store_true",
                    help="also use direct JustWatch data from fetch_justwatch.py (off by default)")
    args = ap.parse_args()

    X = sparse.load_npz(BUILD / "vectors.npz")
    shows = json.loads((BUILD / "shows.json").read_text())
    n = len(shows)
    ids = np.array([s["id"] for s in shows])
    print(f"Mapping {n} shows")

    # --- reduce + neighbours
    k = max(2, min(50, X.shape[1] - 1, n - 1))
    Z = normalize(TruncatedSVD(k, random_state=42).fit_transform(X))
    nn = NearestNeighbors(n_neighbors=min(SIMILAR + 1, n), metric="cosine").fit(Z)
    dist, knn = nn.kneighbors(Z)
    pairs = [[(int(j), 1.0 - float(d)) for j, d in zip(row, drow) if j != i][:SIMILAR]
             for i, (row, drow) in enumerate(zip(knn, dist))]
    similar = [[j for j, _ in pr] for pr in pairs]

    # --- 2-D layout seeded by the previous run
    con = sqlite3.connect(args.db)
    con.execute("CREATE TABLE IF NOT EXISTS layout (id INTEGER PRIMARY KEY, x REAL, y REAL)")
    prev = {} if args.fresh else {i: (x, y) for i, x, y in con.execute("SELECT id, x, y FROM layout")}
    known = np.array([i in prev for i in ids])
    if known.sum() >= 0.5 * n:
        print(f"Seeding layout from previous run ({known.sum()} of {n} shows known)")
        init = np.zeros((n, 2))
        init[known] = [prev[i] for i in ids[known]]
        rng = np.random.default_rng(0)
        centre = init[known].mean(0)
        for i in np.where(~known)[0]:
            nb = [j for j in similar[i] if known[j]]
            init[i] = init[nb].mean(0) if nb else centre
            init[i] += rng.normal(0, 0.05, 2)
        xy = umap_fit(Z, 2, init=init, min_dist=0.08, gentle=True)
        r, scale, mu_n, mu_o = procrustes_align(xy[known], init[known])
        xy = (xy - mu_n) @ r * scale + mu_o
    else:
        print("No usable previous layout; building a fresh one")
        xy = umap_fit(Z, 2, min_dist=0.08, seed=42)

    # --- clusters (on a 10-D embedding: keeps more structure than 2-D)
    print("Clustering")
    Z10 = umap_fit(Z, min(10, n - 2), min_dist=0.0, seed=42)
    labels = HDBSCAN(min_cluster_size=max(12, n // 250), min_samples=8).fit_predict(Z10)
    for _ in range(3):  # let noise points join their neighbours' cluster
        for i in np.where(labels == -1)[0]:
            votes = Counter(labels[j] for j in similar[i] if labels[j] != -1)
            if votes:
                labels[i] = votes.most_common(1)[0][0]
    if (labels == -1).all():
        labels[:] = 0
    cluster_ids = sorted(set(labels) - {-1})
    centroids = {c: np.median(xy[labels == c], axis=0) for c in cluster_ids}
    if (labels == -1).any():
        cids = np.array(cluster_ids)
        cxy = np.array([centroids[c] for c in cluster_ids])
        for i in np.where(labels == -1)[0]:
            labels[i] = cids[np.argmin(((cxy - xy[i]) ** 2).sum(1))]
    remap = {c: k for k, c in enumerate(cluster_ids)}
    labels = np.array([remap[c] for c in labels])
    nclus = len(cluster_ids)
    keys = stable_region_keys(con, ids, labels, nclus)
    print(f"{nclus} clusters")

    # --- normalise coordinates to a 0..1000 square
    lo, hi = xy.min(0), xy.max(0)
    s = 1000 / max(hi - lo)
    pos = (xy - lo) * s + (1000 - (hi - lo) * s) / 2

    # --- cluster names: terms most over-represented in each cluster
    notes = load_notes()
    cands = []
    term_sets = [label_terms(sh) for sh in shows]
    gdf = Counter(t for ts in term_sets for t in ts)
    clusters = []
    for c in range(nclus):
        members = np.where(labels == c)[0]
        cnt = Counter(t for i in members for t in term_sets[i])
        scored = []
        for t, m in cnt.items():
            share = m / len(members)
            if share < 0.2 or gdf[t] < 3:
                continue
            scored.append((share * math.log(share / (gdf[t] / n)), t))
        scored.sort(reverse=True)
        names = []
        for _, t in scored:
            p = pretty(t)
            if all(p.lower() not in q.lower() and q.lower() not in p.lower() for q in names):
                names.append(p)
            if len(names) == 4:  # two make the name; the rest tell apart regions that share one
                break
        cands.append(names)
        cxy = np.median(pos[members], axis=0)
        auto = " · ".join(names[:2]) or f"Region {keys[c]}"
        note = notes.get(str(keys[c]), {})
        clusters.append({"key": keys[c], "label": note.get("name") or auto, "auto": auto,
                         "blurb": note.get("blurb", ""),
                         "x": round(float(cxy[0]), 1), "y": round(float(cxy[1]), 1),
                         "size": int(len(members)), **region_stats(shows, members)})

    # --- regions with the same automatic name get a third (then fourth) term,
    #     then their main country, so no two read alike in guides and lists
    for depth in (3, 4):
        seen = Counter(k["auto"] for k in clusters)
        for k, cand in zip(clusters, cands):
            if seen[k["auto"]] > 1 and len(cand) >= depth:
                k["auto"] = " · ".join(cand[:depth])
    seen = Counter(k["auto"] for k in clusters)
    for k in clusters:
        if seen[k["auto"]] > 1 and k["countries"]:
            k["auto"] += f" ({k['countries'][0][0]})"
    for c, k in enumerate(clusters):
        if not notes.get(str(keys[c]), {}).get("name"):
            k["label"] = k["auto"]
    left = [a for a, m in Counter(k["label"] for k in clusters).items() if m > 1]
    if left:
        print(f"Regions still sharing a name: {left}")

    # --- colours: golden-angle hues, blended with spatial neighbours
    golden = 0.61803398875
    # hue from the stable region key, so a region keeps its colour week to week
    base = np.array([colorsys.hls_to_rgb((keys[c] * golden) % 1.0, 0.6, 0.62) for c in range(nclus)])
    for c in range(nclus):
        clusters[c]["color"] = hsl_hex((keys[c] * golden) % 1.0, 0.62, 0.6)
    tree = cKDTree(pos)
    _, near = tree.query(pos, k=min(10, n))
    rgb = 0.5 * base[labels] + 0.5 * base[labels[near]].mean(1)
    cols = ["#%02x%02x%02x" % tuple(int(round(v * 255)) for v in row) for row in rgb]

    # --- filter vocabularies
    def vocab(fn, min_count=1, limit=None):
        cnt = Counter(t for sh in shows for t in fn(sh))
        return [t for t, m in cnt.most_common(limit) if m >= min_count]

    genres = vocab(lambda s: s["genres"])
    types = vocab(lambda s: [s["type"]] if s["type"] else [])
    langs = vocab(lambda s: [s["original_language"]] if s["original_language"] else [])
    networks = vocab(lambda s: s["networks"], 3)
    keywords = vocab(lambda s: s["keywords"], 3, 1000)
    statuses = vocab(lambda s: [s["status"]] if s["status"] else [])
    countries = vocab(lambda s: s["origin_country"] or [])
    gi, ti, li = ({v: i for i, v in enumerate(x)} for x in (genres, types, langs))
    ni, ki, si = ({v: i for i, v in enumerate(x)} for x in (networks, keywords, statuses))
    ci = {v: i for i, v in enumerate(countries)}

    cols_out = {
        "id": ids.tolist(),
        "name": [sh["name"] for sh in shows],
        "x": [round(float(v), 2) for v in pos[:, 0]],
        "y": [round(float(v), 2) for v in pos[:, 1]],
        "col": cols,
        "cl": labels.tolist(),
        "g": [sum(1 << gi[g] for g in sh["genres"] if g in gi) for sh in shows],
        "t": [ti.get(sh["type"], -1) for sh in shows],
        "lang": [li.get(sh["original_language"], -1) for sh in shows],
        "net": [[ni[x] for x in sh["networks"] if x in ni] for sh in shows],
        "kw": [[ki[x] for x in sh["keywords"] if x in ki] for sh in shows],
        "st": [si.get(sh["status"], -1) for sh in shows],
        "y0": [year(sh["first_air_date"]) for sh in shows],
        "y1": [year(sh["last_air_date"]) or year(sh["first_air_date"]) for sh in shows],
        "va": [round(sh["vote_average"], 1) if sh["vote_count"] >= 5 else 0 for sh in shows],
        "vc": [sh["vote_count"] for sh in shows],
        "h": [total_hours(sh)[0] for sh in shows],
        "on": [sh["original_name"] if sh["original_name"] != sh["name"] else "" for sh in shows],
        "oc": [[ci[c] for c in (sh["origin_country"] or []) if c in ci] for sh in shows],  # origin countries
        "ns": [sh["number_of_seasons"] or 0 for sh in shows],                               # seasons, for dot size
    }
    updated = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    points = {
        "updated": updated, "count": n, "chunks": DETAIL_CHUNKS,
        "genres": genres, "types": types, "languages": langs, "networks": networks,
        "keywords": keywords, "statuses": statuses, "countries": countries, "clusters": clusters, "shows": cols_out,
    }

    # --- write
    for sub in ("details", "providers"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
        (OUT / sub).mkdir(parents=True)
    wtw = where_to_watch(con, shows, use_jw=args.justwatch)
    chunks = [{} for _ in range(DETAIL_CHUNKS)]
    for i, sh in enumerate(shows):
        mins, guess = runtime(sh)
        chunks[sh["id"] % DETAIL_CHUNKS][sh["id"]] = {
            "on": sh["original_name"] if sh["original_name"] != sh["name"] else "",
            "o": sh["overview"], "p": sh["poster_path"],
            "n": sh["networks"], "cr": sh["created_by"], "ca": sh["cast"][:6],
            "s": sh["number_of_seasons"], "e": sh["number_of_episodes"],
            "rt": mins, "hrs": total_hours(sh)[0], "est": guess,
            "kw": sh["keywords"][:20], "imdb": sh.get("imdb_id", ""),
            "st": sh["status"], "fa": sh["first_air_date"], "la": sh["last_air_date"],
            "va": sh["vote_average"], "vc": sh["vote_count"], "r": sh["content_rating"],
            "oc": sh["origin_country"], "g": sh["genres"], "ty": sh["type"],
            "sim": [int(ids[j]) for j in similar[i]],
            # where to watch: source (j = JustWatch, t = TMDB, "" = none),
            # countries with any offer, JustWatch page
            "ws": wtw[sh["id"]][1], "wc": len(wtw[sh["id"]][0]), "jw": wtw[sh["id"]][2],
        }
    for c, d in enumerate(chunks):
        (OUT / "details" / f"{c:02d}.json").write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
    countries = write_providers(con, shows, wtw)
    write_graph_and_vectors(pairs, Z, points)
    # IMDb ids in points order, loaded only when someone imports an IMDb export
    (OUT / "ids.json").write_text(json.dumps({"imdb": [sh.get("imdb_id") or "" for sh in shows]}, separators=(",", ":")))
    (OUT / "points.json").write_text(json.dumps(points, ensure_ascii=False, separators=(",", ":")))
    status = {"updated": updated, "shows": n, "clusters": nclus}
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    con.execute("DELETE FROM layout")
    con.executemany("INSERT INTO layout (id, x, y) VALUES (?,?,?)",
                    [(int(i), float(x), float(y)) for i, (x, y) in zip(ids, xy)])
    con.commit()
    size = (OUT / "points.json").stat().st_size / 1e6
    print(f"Wrote {OUT}/points.json ({size:.1f} MB), {DETAIL_CHUNKS} detail chunks, "
          f"where-to-watch for {countries} countries")


def stable_region_keys(con, ids, labels, nclus) -> list[int]:
    """Give each cluster a key that survives re-clustering.

    A new cluster inherits the key of the previous run's region it overlaps
    most (Jaccard >= 0.25, best overlaps first); otherwise it gets a new key.
    Keys are never reused, so hand-written notes can't jump to another region.
    """
    con.execute("CREATE TABLE IF NOT EXISTS regions (id INTEGER PRIMARY KEY, region INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    prev = dict(con.execute("SELECT id, region FROM regions"))
    prev_size = Counter(prev.values())
    row = con.execute("SELECT v FROM meta WHERE k='next_region'").fetchone()
    next_key = max(int(row[0]) if row else 1, max(prev_size, default=0) + 1)
    cand = []
    for c in range(nclus):
        members = ids[labels == c]
        for key, m in Counter(prev[i] for i in members if i in prev).items():
            jac = m / (len(members) + prev_size[key] - m)
            if jac >= 0.25:
                cand.append((jac, c, key))
    keys: dict[int, int] = {}
    used = set()
    for jac, c, key in sorted(cand, reverse=True):
        if c not in keys and key not in used:
            keys[c] = key
            used.add(key)
    kept = len(keys)
    for c in range(nclus):
        if c not in keys:
            keys[c] = next_key
            next_key += 1
    con.execute("DELETE FROM regions")
    con.executemany("INSERT INTO regions (id, region) VALUES (?, ?)",
                    [(int(i), keys[int(c)]) for i, c in zip(ids, labels)])
    con.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('next_region', ?)", (str(next_key),))
    con.commit()
    if prev:
        print(f"Regions: {kept} kept their key from last run, {nclus - kept} new")
    return [keys[c] for c in range(nclus)]


def load_notes() -> dict:
    """pipeline/cluster_notes.json: {"12": {"name": "Nordic noir", "blurb": "..."}}."""
    try:
        raw = json.loads(NOTES.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Ignoring {NOTES.name}: not valid JSON ({e})")
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict) and not str(k).startswith("_")}


def region_stats(shows, members) -> dict:
    """What a region guide shows: top shows and its dominant traits.

    Shows are in popularity order, so the lowest indices are the best known.
    Shares are fractions of the region's shows.
    """
    m = len(members)
    top = [shows[i]["id"] for i in sorted(members)[:5]]
    def share(fn, k=3):
        cnt = Counter(v for i in members for v in fn(shows[i]))
        return [[v, round(c / m, 2)] for v, c in cnt.most_common(k)]
    yrs = sorted(year(shows[i]["first_air_date"]) for i in members if year(shows[i]["first_air_date"]))
    span = [yrs[len(yrs) // 10], yrs[len(yrs) // 2], yrs[(len(yrs) * 9) // 10]] if yrs else []
    return {
        "top": top,
        "genres": share(lambda s: s["genres"]),
        "networks": share(lambda s: s["networks"]),
        "countries": share(lambda s: s["origin_country"] or []),
        "langs": share(lambda s: [s["original_language"]] if s["original_language"] else [], 2),
        "years": span,  # 10th percentile, median, 90th percentile first-air year
    }


def write_graph_and_vectors(pairs, Z, points):
    """Binary files the browser loads only when needed.

    graph.bin   - neighbour indices (uint16, or uint32 for 65,536+ shows), N x K,
                  then similarities as uint8 (0..255 = cosine 0..1), N x K.
                  Used by the bridge (shortest path) feature.
    vectors.bin - each show's 50-D vector as int8 (x127), N x D, in points
                  order. Used for compare scores and, later, "For you".
    """
    n, k = len(pairs), SIMILAR
    wide = n >= 65536
    idx = np.zeros((n, k), dtype=np.uint32 if wide else np.uint16)
    sim = np.zeros((n, k), dtype=np.uint8)
    for i, pr in enumerate(pairs):
        for t in range(k):
            j, w = pr[t] if t < len(pr) else pr[-1]  # pad short rows by repeating
            idx[i, t] = j
            sim[i, t] = int(round(max(0.0, min(1.0, w)) * 255))
    (OUT / "graph.bin").write_bytes(idx.tobytes() + sim.tobytes())
    q = np.clip(np.round(Z * 127), -127, 127).astype(np.int8)
    (OUT / "vectors.bin").write_bytes(q.tobytes())
    points["graph"] = {"k": k, "idx": "u32" if wide else "u16"}
    # typical similarity of two random shows: the site maps this to "0% alike"
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    base = float(np.median((Z[a] * Z[b]).sum(1)))
    points["vectors"] = {"dims": int(Z.shape[1]), "scale": 127, "base": round(base, 4)}


def where_to_watch(con, shows, use_jw=False) -> dict:
    """{show id: (offers by country, source, JustWatch url)}.

    Order of preference, per show:
      1. fresh direct JustWatch data (it is complete: a missing country means
         the show isn't offered there)
      2. TMDB's where-to-watch, refreshed with every TMDB fetch
      3. older direct JustWatch data (better than nothing)
      4. nothing - the site says so and links to a JustWatch search
    Without use_jw only 2 and 4 apply.
    """
    jw = {}
    try:
        if not use_jw:
            raise sqlite3.OperationalError("direct JustWatch data not wanted")
        fresh = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=JW_FRESH_DAYS)).isoformat(timespec="seconds")
        for sid, url, offers, at in con.execute(
                "SELECT id, url, offers, offers_at FROM jw WHERE status IN ('ok','manual')"):
            data = json.loads(offers) if offers else None
            jw[sid] = (data if data is not None and (at or "") >= fresh else None, url or "", data or {})
    except sqlite3.OperationalError:
        pass  # JustWatch step has never run
    out = {}
    used = Counter()
    for sh in shows:
        direct, url, old = jw.get(sh["id"], (None, "", {}))
        tmdb = {cc: o for cc, o in (sh.get("providers") or {}).items() if o}
        if direct is not None and (direct or not tmdb):
            out[sh["id"]] = (direct, "j" if direct else "", url)
        elif tmdb:
            out[sh["id"]] = (tmdb, "t", url)
        elif old:
            out[sh["id"]] = (old, "j", url)
        else:
            out[sh["id"]] = ({}, "", url)
        used[out[sh["id"]][1] or "none"] += 1
    print(f"Where to watch: {used['j']} from JustWatch, {used['t']} from TMDB, {used['none']} with no data")
    return out


def write_providers(con, shows, wtw) -> int:
    """One file per country: {"top": [provider ids by #shows], "shows": {id: {s,f,r,b}}}."""
    by_cc: dict[str, dict] = {}
    for sh in shows:
        for cc, offers in wtw[sh["id"]][0].items():
            if not offers:
                continue
            by_cc.setdefault(cc, {})[str(sh["id"])] = offers
    names = {}
    try:
        names = {pid: (name, logo, prio) for pid, name, logo, prio in con.execute(
            "SELECT id, name, logo, priority FROM providers")}
    except sqlite3.OperationalError:
        pass  # cache predates where-to-watch
    used = set()
    counts = {}
    for cc, entries in sorted(by_cc.items()):
        cnt = Counter(pid for o in entries.values() for kind in ("s", "f") for pid in o.get(kind, []))
        top = [pid for pid, _ in cnt.most_common()]
        used.update(pid for o in entries.values() for ids in o.values() for pid in ids)
        (OUT / "providers" / f"{cc}.json").write_text(
            json.dumps({"top": top, "shows": entries}, separators=(",", ":")))
        counts[cc] = len(entries)
    index = {
        "countries": counts,
        "providers": {str(p): [names.get(p, (f"Provider {p}", "", 999))[0], names.get(p, ("", "", 999))[1]]
                      for p in sorted(used)},
    }
    (OUT / "providers" / "index.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")))
    return len(counts)


if __name__ == "__main__":
    main()
