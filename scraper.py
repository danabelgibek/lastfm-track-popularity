"""
scraper.py - Last.fm data collection for SDS307 HW1 (Option 2).

Stages
  1. Candidate tracks    : tag.getTopTracks  -> lastfm_candidates.csv
  2. Track information   : track.getInfo     -> lastfm_track_info.jsonl   (resumable)
  3. Artist information  : artist.getInfo    -> lastfm_artist_info.jsonl  (resumable)
  4. Dataset building    : merge + feature engineering -> lastfm_tracks_raw_full.csv

Usage
  export LASTFM_API_KEY="your_key"          # never hardcode the key in the code
  export LASTFM_USER_AGENT="YourProject/1.0"  # optional
  python scraper.py

Rules followed: public API only, >= 1 second between requests, no personal data.
Collection date is stored in the column 'collected_at'.
"""
import requests, time, re, os, json
import numpy as np
import pandas as pd

# ==========================================
# CONFIG
# ==========================================
API_KEY = os.environ.get("LASTFM_API_KEY")
if not API_KEY:
    raise SystemExit("Set the LASTFM_API_KEY environment variable first.")
USER_AGENT = os.environ.get("LASTFM_USER_AGENT", "UniversityRegressionProject/1.0")
BASE_URL = "https://ws.audioscrobbler.com/2.0/"
SLEEP = 1.0                      # course requirement: >= 1 second between requests
TARGET_CANDIDATES = 12000        # collect extra candidates (some tracks are lost later)

RAW_FILE   = "lastfm_tracks_raw.csv"     # optional: first-pass data (seed candidates)
CAND_FILE  = "lastfm_candidates.csv"
TRACK_FILE = "lastfm_track_info.jsonl"
ART_FILE   = "lastfm_artist_info.jsonl"
FINAL_FILE = "lastfm_tracks_raw_full.csv"   # raw dataset = input of cleaning.py

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def lastfm_get(method, max_retries=4, **params):
    """Single API call with a pause, retries and a check for errors inside the JSON."""
    p = {"method": method, "api_key": API_KEY, "format": "json", **params}
    for attempt in range(max_retries):
        try:
            r = session.get(BASE_URL, params=p, timeout=15)
            time.sleep(SLEEP)
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1))
                continue
            data = r.json()
            if "error" in data:
                if data["error"] in (8, 11, 16, 29):   # temporary errors / rate limit
                    time.sleep(5 * (attempt + 1))
                    continue
                return None                             # e.g. 6 = not found
            return data
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def key(artist, name):
    return f"{artist.strip().lower()} - {name.strip().lower()}"


def done_keys(path):
    """Keys already written to a .jsonl file (used to resume after an interruption)."""
    s = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                s.add(json.loads(line)["_key"])
    return s


def as_list(x):
    return [x] if isinstance(x, dict) else (x or [])


# ==========================================
# STAGE 1. Candidates: tag.getTopTracks
# ==========================================
NEW_TAGS = ["indie", "alternative", "electronic", "hip-hop", "rock", "jazz", "metal",
            "folk", "punk", "soul", "r&b", "reggae", "country", "blues", "classical",
            "ambient", "house", "techno", "trap", "k-pop", "latin", "funk", "disco",
            "shoegaze", "synthpop", "post-rock", "lo-fi", "singer-songwriter"]

if os.path.exists(CAND_FILE):
    # Resume: the candidate list was already collected
    cand_df = pd.read_csv(CAND_FILE, dtype=str, keep_default_na=False)
    cand_df["rank_in_genre"] = cand_df["rank_in_genre"].astype(int)
    candidates = cand_df.to_dict("records")
    print(f"Loaded {len(candidates)} candidates from {CAND_FILE}")
else:
    candidates = []
    seen = set()
    if os.path.exists(RAW_FILE):
        old = pd.read_csv(RAW_FILE)
        for _, r in old.iterrows():
            k = key(str(r["artist_name"]), str(r["track_name"]))
            if k not in seen:
                seen.add(k)
                candidates.append({"track_name": r["track_name"], "artist_name": r["artist_name"],
                                   "genre_tag": r["genre_tag"], "rank_in_genre": r["rank_in_genre"]})
    print(f"Loaded from the first-pass file: {len(candidates)} candidates")

    for tag in NEW_TAGS:
        if len(candidates) >= TARGET_CANDIDATES:
            break
        print(f"Collecting tag '{tag}' (total candidates so far: {len(candidates)})...")
        for page in range(1, 21):
            if len(candidates) >= TARGET_CANDIDATES:
                break
            data = lastfm_get("tag.gettoptracks", tag=tag, limit=50, page=page)
            tracks = (data or {}).get("tracks", {}).get("track", [])
            if not tracks:
                break
            for t in tracks:
                name, artist = t.get("name"), (t.get("artist") or {}).get("name")
                if not name or not artist:
                    continue
                k = key(artist, name)
                if k in seen:
                    continue
                seen.add(k)
                candidates.append({"track_name": name, "artist_name": artist, "genre_tag": tag,
                                   "rank_in_genre": int((t.get("@attr") or {}).get("rank", 0))})

    pd.DataFrame(candidates).to_csv(CAND_FILE, index=False)
    print(f"Stage 1 finished: {len(candidates)} candidate tracks saved to {CAND_FILE}")


# ==========================================
# STAGE 2. track.getInfo (resumable)
# ==========================================
def parse_track(c, data):
    t = data.get("track", {})
    album = t.get("album") or {}
    tags = as_list((t.get("toptags") or {}).get("tag"))
    wiki = (t.get("wiki") or {}).get("summary", "") or ""
    return {
        "_key": key(c["artist_name"], c["track_name"]),
        "listeners_track": int(t.get("listeners", 0) or 0),
        "playcount": int(t.get("playcount", 0) or 0),                   # TARGET
        "duration_ms": int(t.get("duration", 0) or 0),
        "has_album": int(bool(album)),
        "album_title": album.get("title", ""),
        "album_position": int((album.get("@attr") or {}).get("position", 0) or 0),
        "n_track_tags": len(tags),
        "top_tag": tags[0]["name"].lower() if tags else "none",
        "has_wiki": int(bool(wiki)),
        "wiki_length": len(wiki),
    }


done = done_keys(TRACK_FILE)
with open(TRACK_FILE, "a", encoding="utf-8") as f:
    for i, c in enumerate(candidates):
        if key(c["artist_name"], c["track_name"]) in done:
            continue
        data = lastfm_get("track.getInfo", artist=c["artist_name"], track=c["track_name"], autocorrect=1)
        if data and "track" in data:
            f.write(json.dumps(parse_track(c, data), ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps({"_key": key(c["artist_name"], c["track_name"]), "_missing": 1}) + "\n")
        if i % 200 == 0:
            f.flush()
            print(f"track.getInfo: {i}/{len(candidates)}")
print("Stage 2 finished: track info saved to", TRACK_FILE)


# ==========================================
# STAGE 3. artist.getInfo (one request per unique artist)
# ==========================================
def parse_artist(name, data):
    a = data.get("artist", {})
    stats = a.get("stats") or {}
    sim = as_list((a.get("similar") or {}).get("artist"))
    tags = as_list((a.get("tags") or {}).get("tag"))
    bio = (a.get("bio") or {}).get("summary", "") or ""
    return {
        "_key": name.lower(),
        "artist_listeners": int(stats.get("listeners", 0) or 0),
        "artist_playcount": int(stats.get("playcount", 0) or 0),
        "n_similar_artists": len(sim),
        "n_artist_tags": len(tags),
        "has_artist_bio": int(bool(bio.strip())),
        "artist_bio_length": len(bio),
        "artist_on_tour": int(a.get("ontour", 0) or 0),
    }


tinfo = pd.read_json(TRACK_FILE, lines=True)
good_keys = set(tinfo.loc[tinfo["_missing"].isna(), "_key"]) if "_missing" in tinfo else set(tinfo["_key"])
artists = sorted({c["artist_name"] for c in candidates
                  if key(c["artist_name"], c["track_name"]) in good_keys})
print(f"Unique artists to process: {len(artists)}")

done_a = done_keys(ART_FILE)
with open(ART_FILE, "a", encoding="utf-8") as f:
    for i, name in enumerate(artists):
        if name.lower() in done_a:
            continue
        data = lastfm_get("artist.getInfo", artist=name, autocorrect=1)
        if data and "artist" in data:
            f.write(json.dumps(parse_artist(name, data), ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps({"_key": name.lower(), "_missing": 1}) + "\n")
        if i % 200 == 0:
            f.flush()
            print(f"artist.getInfo: {i}/{len(artists)}")
print("Stage 3 finished: artist info saved to", ART_FILE)


# ==========================================
# STAGE 4. Build the dataset + feature engineering
# ==========================================
cand = pd.DataFrame(candidates)
cand["_key"] = [key(a, n) for a, n in zip(cand.artist_name, cand.track_name)]
cand["_akey"] = cand.artist_name.str.lower()

ti = pd.read_json(TRACK_FILE, lines=True)
ai = pd.read_json(ART_FILE, lines=True).rename(columns={"_key": "_akey"})
df = (cand.merge(ti, on="_key", how="inner")
          .merge(ai, on="_akey", how="inner"))
df = df[df.get("_missing_x").isna() & df.get("_missing_y").isna()] if "_missing_x" in df else df
df = df.drop(columns=[c for c in df.columns if c.startswith("_missing")])
df = df[(df.playcount > 0) & (df.artist_listeners > 0)].drop_duplicates("_key")


def flag(pattern, s):
    return int(bool(re.search(pattern, s.lower())))


n, a = df.track_name, df.artist_name
df["name_length"]       = n.str.len()
df["artist_length"]     = a.str.len()
df["word_count_name"]   = n.str.split().str.len()
df["artist_word_count"] = a.str.split().str.len()
df["is_featured"] = n.apply(lambda s: flag(r"\b(feat|ft|featuring)\b", s))
df["has_remix"]   = n.apply(lambda s: flag(r"\bremix\b", s))
df["has_live"]    = n.apply(lambda s: flag(r"\blive\b", s))
df["has_edit"]    = n.apply(lambda s: flag(r"\bedit\b", s))
df["has_version"] = n.apply(lambda s: flag(r"\bversion\b", s))
df["name_has_paren"] = n.str.contains(r"[\(\[]").astype(int)
df["name_has_digit"] = n.str.contains(r"\d").astype(int)
df["name_is_ascii"]  = n.apply(lambda s: int(s.isascii()))
df["artist_is_collab"] = a.str.contains(r"&|,| feat| and | x ", case=False, regex=True).astype(int)
df["name_equals_album"] = (n.str.lower().str.strip() == df.album_title.str.lower().str.strip()).astype(int)
df["duration_min"] = df.duration_ms.replace(0, np.nan) / 60000
df["artist_plays_per_listener"] = df.artist_playcount / df.artist_listeners
df["log_artist_listeners"] = np.log1p(df.artist_listeners)

FEATURES = ["genre_tag", "rank_in_genre", "name_length", "artist_length", "word_count_name",
            "artist_word_count", "is_featured", "has_remix", "has_live", "has_edit", "has_version",
            "name_has_paren", "name_has_digit", "name_is_ascii", "artist_is_collab", "name_equals_album",
            "duration_min", "has_album", "album_position", "n_track_tags", "top_tag", "has_wiki", "wiki_length",
            "artist_listeners", "artist_playcount", "n_similar_artists", "n_artist_tags",
            "has_artist_bio", "artist_bio_length", "artist_on_tour",
            "artist_plays_per_listener", "log_artist_listeners"]
TARGET = "playcount"
IDS = ["track_name", "artist_name"]

out = df[IDS + FEATURES + [TARGET]].copy()
out["collected_at"] = pd.Timestamp.today().strftime("%Y-%m-%d")
out.to_csv(FINAL_FILE, index=False)
print(f"Rows: {len(out)} | Features: {len(FEATURES)} (excluding IDs and target)")
print(f"Saved to: {FINAL_FILE}")
