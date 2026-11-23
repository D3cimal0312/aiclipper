#!/usr/bin/env python3
"""
Local AI clipper: long video -> vertical (9:16) short clips with word-by-word captions.
Everything runs on your own machine:
  Whisper (faster-whisper)  -> transcript with word timestamps
  Ollama (local LLM)        -> picks the most viral-worthy moments
  OpenCV                    -> finds the speaker's face to keep them in frame
  ffmpeg                    -> cuts, crops to 9:16, burns captions

Usage:
  python clipper.py my_video.mp4
  python clipper.py my_video.mp4 --clips 8 --min 20 --max 55 --llm qwen2.5:7b --whisper small
"""
import argparse
import base64
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import requests
from faster_whisper import WhisperModel

OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPT = """You are an expert short-form video editor (TikTok, Reels, YouTube Shorts).
Below is a timestamped transcript. Pick up to {n} moments that would make the most engaging standalone clips.

Rules:
- Each clip must be between {mn} and {mx} seconds long.
- Start right on the hook: the funniest or most curious line, a bold claim, or a question. No greetings, no slow lead-in, no setup that takes more than a few seconds.
- End on the punchline or payoff, right after it lands. A clip that ends on its best line loops cleanly and gets rewatched. Never cut mid-sentence.
- The clip must make sense without the rest of the video.
- Use the exact timestamps shown in the transcript.
- score is 1-100 for how likely viewers are to watch to the end and rewatch. Be critical; most moments are not above 70.{guidance}

Respond ONLY with JSON in this exact shape:
{{"clips":[{{"start":<seconds>,"end":<seconds>,"score":<int>,"title":"<hook title, max 8 words, plain text, no emojis or quotes>","reason":"<why it works>"}}]}}

TRANSCRIPT:
{transcript}
"""


# ---------- 1. Transcribe ----------
def transcribe(video, model_size, device, compute_type, language):
    print(f"[1/4] Transcribing with Whisper '{model_size}' (first run downloads the model)...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(video), word_timestamps=True, language=language, vad_filter=True
    )
    out = []
    for s in segments:
        words = [
            {"w": w.word.strip(), "start": w.start, "end": w.end}
            for w in (s.words or [])
            if w.word.strip()
        ]
        out.append({"start": s.start, "end": s.end, "text": s.text.strip(), "words": words})
    print(f"      {len(out)} segments, detected language: {info.language}")
    return out


# ---------- 2. Pick highlights with a local LLM ----------
def chunk_segments(segments, window):
    chunks, cur, t0 = [], [], None
    for s in segments:
        if t0 is None:
            t0 = s["start"]
        cur.append(s)
        if s["end"] - t0 >= window:
            chunks.append(cur)
            cur, t0 = [], None
    if cur:
        chunks.append(cur)
    return chunks


def ask_llm(model, prompt):
    r = None
    for ctx in (8192, 4096):  # a 500 from Ollama is usually out-of-memory: retry once with a smaller context
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": ctx},
            },
            timeout=900,
        )
        if r.status_code < 500:
            break
        print(f"      Ollama error {r.status_code}: {r.text[:200]} (retrying with a smaller context)")
    if r.status_code >= 500:
        sys.exit(f"Ollama failed ({r.text[:200]}). Most likely '{model}' does not fit in memory. "
                 "Close other apps, restart Ollama (right-click tray icon > Quit, open again), "
                 "or use a smaller model: -Llm qwen2.5:7b")
    r.raise_for_status()
    content = r.json()["message"]["content"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    clips = []
    for c in data.get("clips", []):
        try:
            clips.append(
                {
                    "start": float(c["start"]),
                    "end": float(c["end"]),
                    "score": int(c.get("score", 50)),
                    "title": str(c.get("title", "")).strip(),
                    "reason": str(c.get("reason", "")).strip(),
                }
            )
        except (KeyError, ValueError, TypeError):
            continue
    return clips


def snap(clip, segments, mn, mx):
    """Snap clip edges to sentence/segment boundaries and enforce length limits."""
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    word_ends = [w["end"] for s in segments for w in s["words"]]
    st = min(starts, key=lambda x: abs(x - clip["start"]))
    en = min(ends, key=lambda x: abs(x - clip["end"]))
    if mn <= en - st <= mx:
        return st, en
    # Pick a sentence end inside the allowed range, closest to what the AI wanted
    ok = [e for e in ends if st + mn <= e <= st + mx]
    if ok:
        return st, min(ok, key=lambda e: abs(e - clip["end"]))
    # No sentence end fits: fall back to the nearest word end inside the range
    ok = [e for e in word_ends if st + mn <= e <= st + mx]
    if ok:
        target = st + mx if en - st > mx else st + mn
        return st, min(ok, key=lambda e: abs(e - target))
    return st, None  # not enough video left to make a clip of this length


LAUGH_RE = re.compile(r"(?i)\b(ha(?:ha)+|lmao|lmfao|bro+|no way|yo+|oh my god|omg|what the|bruh|let'?s go+|chat)\b")


def rank_chunks(chunks, video, k):
    """Keep the k transcript windows that look most hyped (shouting, laughing, loud audio)."""
    a = audio_levels(video, 1.0)
    z = _robust_z(20 * np.log10(a + 1e-5), 2.0) if len(a) else None
    tx, au = [], []
    for ch in chunks:
        text = " ".join(s["text"] for s in ch)
        t0, t1 = ch[0]["start"], ch[-1]["end"]
        mins = max((t1 - t0) / 60, 0.5)
        tx.append((text.count("!") + 0.7 * len(re.findall(r"\b[A-Z]{3,}\b", text))
                   + 1.5 * len(LAUGH_RE.findall(text))) / mins)
        v = 0.0
        if z is not None:
            seg = np.sort(z[int(t0) : int(t1) + 1])
            if len(seg):
                v = float(seg[-max(1, len(seg) // 10):].mean())
        au.append(v)

    def norm(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / (x.std() + 1e-6)

    score = norm(tx) + norm(au)
    keep = sorted(np.argsort(-score)[:k])
    return [chunks[i] for i in keep]


def overlaps(a, b, ranges):
    return any(a < hi and b > lo for lo, hi in ranges)


def pick_clips(segments, llm, n, mn, mx, window, guidance="", skip=(), prefilter=0, video=None):
    print(f"[2/4] Asking local LLM '{llm}' to find the best moments...")
    chunks = chunk_segments(segments, window)
    if prefilter and len(chunks) > prefilter and video is not None:
        print(f"      Long video: checking only the {prefilter} most hyped of {len(chunks)} windows.")
        chunks = rank_chunks(chunks, video, prefilter)
    g = ("\n" + guidance.strip()) if guidance and guidance.strip() else ""
    candidates = []
    for i, chunk in enumerate(chunks, 1):
        transcript = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in chunk)
        prompt = PROMPT.format(n=5, mn=mn, mx=mx, guidance=g, transcript=transcript)
        found = ask_llm(llm, prompt)
        print(f"      chunk {i}/{len(chunks)}: {len(found)} candidates")
        candidates.extend(found)

    snapped = []
    for c in candidates:
        st, en = snap(c, segments, mn, mx)
        if en is None:
            continue
        c["start"], c["end"] = st, en
        snapped.append(c)
    # strict length check: no clip outside the min/max range
    candidates = [c for c in snapped if mn - 0.01 <= c["end"] - c["start"] <= mx + 0.01]
    if skip:
        candidates = [c for c in candidates if not overlaps(c["start"], c["end"], skip)]
    candidates.sort(key=lambda c: c["score"], reverse=True)

    chosen = []
    for c in candidates:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    chosen.sort(key=lambda c: c["start"])
    if len(chosen) < n:
        print(f"      Only {len(chosen)} clips fit your length range ({mn}-{mx}s). "
              f"Try a wider -Min/-Max or a longer video.")
    return chosen


# ---------- 2b. YouTube metadata: hook title, description, tags ----------
DEFAULT_PROFILE = {
    "show": "",
    "hashtag": "",
    "characters": [],
    "description_keywords": "",
    "base_tags": [],
    "mandatory_line": "",
    "emojis": ["\U0001F602", "\U0001F480"],  # laughing face, skull
    # campaign / platform extras (all optional)
    "required_name": "",          # must appear in overlay, YouTube title, TikTok caption
    "required_hashtag": "",       # must appear in YouTube description and TikTok caption
    "context": "",                # one phrase describing the source, used in the AI prompt
    "pick_guidance": "",          # extra rules for choosing clips
    "tone_rules": "",             # extra rules for the AI titles/summaries
    "banned_words": [],           # never allowed in generated text
    "extra_hashtags": {"youtube": [], "tiktok": []},
    "source_credit": "",
    "mandatory_line_required": True,
    "category": "Entertainment",
    "fallback_title": "",
    "audience_utc_offset": None,  # e.g. -5 = US Eastern; post times are in this zone
    "defaults": {},               # clips, min, max, language, layout, facecam, post_times, prefilter
}
MISSING_LINE = "[[MANDATORY LINE MISSING: set mandatory_line in channel.json]]"

META_PROMPT = """You write YouTube Shorts metadata for short entertainment clips@@SHOWLINE@@.
Read the clip transcript and respond ONLY with JSON in exactly this shape:
{"titles":["...","...","..."],"caption":"...","emoji":"...","characters":["..."],"summary":"...","question":"...","tags":["..."]}

Rules:
- caption: big on-screen text for the first 2 seconds, 3 to 7 words, a reaction line about what is happening (example style: "Kai Cenat can't believe this"). It must be true to the clip, positive or neutral, in @@LANG@@.
- titles: 3 different hook titles, 4 to 9 words each, written in @@LANG@@. Create curiosity or tension so people must watch to the end.
  Put the character name (if any) and the key hook words in the FIRST 40 characters.
  Do NOT copy 3 or more words in a row from the transcript. Do NOT reveal the punchline. Must be true to the clip.
  No emojis, hashtags or quotation marks.
- emoji: the single best fit from this list: @@EMOJIS@@
- characters: only names from this list that clearly appear in or are part of the clip: @@CHARACTERS@@. Use [] if none or unsure.
- summary: one sentence (max 25 words) saying what happens in the clip. No hashtags.
- question: one short question for viewers that invites comments about this clip.
- tags: 6 to 10 search phrases (1 to 3 words each) that viewers might type to find this clip.
@@EXTRA@@

Style examples of titles (other topics): "Why They Should've Hired The Original Janitor", "Nobody Warns You About This", "The Mistake Costing You Thousands".

TRANSCRIPT:
@@TEXT@@
"""


def ask_json(model, prompt, temperature=0.8):
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096},
        },
        timeout=900,  # a model partly running from RAM can be slow
    )
    r.raise_for_status()
    content = r.json()["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


def _words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


CJK_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]")


def has_cjk(s):
    return bool(CJK_RE.search(str(s)))


def banned_hits(s, words):
    low = str(s).lower()
    return [w for w in (str(x).lower().strip() for x in (words or []))
            if w and re.search(r"\b" + re.escape(w) + r"\b", low)]


def with_name(text, name):
    text = (text or "").strip()
    if not name or name.lower() in text.lower():
        return text
    first = name.split()[0]
    if text.lower().startswith(first.lower() + " "):      # "Kai can't believe this" -> "Kai Cenat can't believe this"
        return name + text[len(first):]
    return f"{name}: {text}" if text else name


def pick_title(candidates, transcript_text, fallback, reject=None):
    """First title that is 4-9 words, doesn't lift 3 words in a row from the clip, and passes `reject`."""
    tw = _words(transcript_text)
    grams = {tuple(tw[i : i + 3]) for i in range(len(tw) - 2)}
    for t in candidates:
        t = re.sub(r"[\"“”{}\\#]", "", str(t)).strip()
        w = _words(t)
        if not 4 <= len(t.split()) <= 9:
            continue
        if any(tuple(w[i : i + 3]) in grams for i in range(len(w) - 2)):
            continue
        if reject and reject(t):
            continue
        return t
    return fallback


def load_profile(path):
    p = dict(DEFAULT_PROFILE)
    if path:
        try:
            p.update(json.loads(Path(path).read_text(encoding="utf-8-sig")))
        except (OSError, json.JSONDecodeError) as ex:
            sys.exit(f"Could not read channel file '{path}': {ex}")
    return p


def clean_text(s):
    s = re.sub(r"#\w+", "", str(s))          # the description may carry only your own hashtag
    s = re.sub(r"[{}\\<>]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_yt_title(hook, emoji, show, limit=55):
    hook = hook.strip()
    if len(hook) > limit:
        hook = hook[:limit].rsplit(" ", 1)[0].rstrip(",;:-") + "…"
    if show and show.lower() in hook.lower():
        show = ""
    return hook + (f" {emoji}" if emoji else "") + (f" | {show}" if show else "")


def hashtags(profile, platform):
    raw = [profile.get("required_hashtag") or profile.get("hashtag") or ""]
    raw += list((profile.get("extra_hashtags") or {}).get(platform, []))
    seen, out = set(), []
    for h in raw:
        h = str(h).strip().lstrip("#")
        if h and h.lower() not in seen:
            seen.add(h.lower())
            out.append("#" + h)
    return out


def build_description(c, profile):
    name = profile["required_name"]
    parts = []
    summary = c.get("summary") or ""
    if name and name.lower() not in summary.lower():
        summary = f"{name}: {summary}" if summary else f"{name} GTA moment."
    if summary:
        parts.append(summary)
    feat = [x for x in c.get("characters", []) if x != name]
    if feat:
        parts.append("Featuring: " + ", ".join(feat))
    if profile["description_keywords"]:
        parts.append(profile["description_keywords"])
    if profile.get("source_credit"):
        parts.append(profile["source_credit"])
    if c.get("question"):
        parts.append(c["question"])
    ml = profile["mandatory_line"].strip()
    if not ml or ml.upper().startswith("PASTE"):
        if profile.get("mandatory_line_required", True):
            parts.append(MISSING_LINE)
    else:
        parts.append(ml)
    tags = hashtags(profile, "youtube")
    if tags:
        parts.append(" ".join(tags))
    return "\n\n".join(parts)


def build_tags(c, profile, limit=480):
    seen, tags, total = set(), [], 0
    for t in list(profile["base_tags"]) + list(c.get("characters", [])) + list(c.get("llm_tags", [])):
        t = clean_text(t).strip(",")
        if not t or t.lower() in seen:
            continue
        if has_cjk(t) or banned_hits(t, profile["banned_words"]):
            continue
        if total + len(t) + 2 > limit:
            break
        seen.add(t.lower())
        tags.append(t)
        total += len(t) + 2
    return tags


def build_tiktok(c, profile, limit=150):
    """TikTok caption (max 150 chars): hook + emoji + hashtags, always with the required name."""
    name = profile["required_name"]
    tail = " ".join(hashtags(profile, "tiktok"))
    emoji = c.get("emoji", "")
    hook = with_name(c["title"], name)
    room = limit - len(tail) - 1 - (len(emoji) + 1 if emoji else 0)
    if len(hook) > room:
        hook = hook[: max(room, 0)].rsplit(" ", 1)[0].rstrip(",;:-")
    cap = " ".join(x for x in (hook, emoji, tail) if x)
    if name and name.lower() not in cap.lower():
        cap = " ".join(x for x in (name, emoji, tail) if x)[:limit]
    return cap


RETRY_NOTE = ("\n\nIMPORTANT: your first titles were rejected because they were too short or copied "
              "words from the clip. Write 3 NEW titles of 5 to 9 words that reword the idea in a "
              "curious, punchy way. Respond with the same JSON shape.")


def _title_list(d):
    t = d.get("titles") if isinstance(d, dict) else None
    return t if isinstance(t, list) else []


def make_metadata(chosen, segments, llm, lang, profile, mode="speech", note=""):
    print("      Writing hook titles, descriptions and tags...")
    emojis = profile["emojis"] or []
    name = profile["required_name"]
    banned = profile["banned_words"]
    bad = lambda t: has_cjk(t) or bool(banned_hits(t, banned))
    extra = profile.get("tone_rules", "").strip()
    if banned:
        extra += ("\n" if extra else "") + "- Never use any of these words: " + ", ".join(banned) + "."
    extra = ("\n" + extra) if extra else ""
    showline = profile["context"] if profile["context"] else (f" for the show \"{profile['show']}\"" if profile["show"] else "")
    norm = lambda n: re.sub(r"[^a-z0-9 ]", "", str(n).lower()).strip()
    known = {norm(n): n for n in profile["characters"]}
    for c in chosen:
        text = " ".join(
            s["text"] for s in segments
            if s["start"] >= c["start"] - 0.5 and s["end"] <= c["end"] + 0.5
        )
        c["spoken_flags"] = banned_hits(text, banned)
        if mode == "gameplay":
            notes = " | ".join(x for x in (note, c.get("notes", "")) if x)
            prompt = (META_PROMPT_GAME
                      .replace("@@SHOW@@", profile["show"] or "a video game")
                      .replace("@@LANG@@", lang)
                      .replace("@@EMOJIS@@", " ".join(emojis) if emojis else "(none, use an empty string)")
                      .replace("@@CHARACTERS@@", ", ".join(profile["characters"]) or "(none)")
                      .replace("@@NOTES@@", notes or "(none: no description available)")
                      .replace("@@TEXT@@", text[:3500] or "(none)"))
        else:
            prompt = (META_PROMPT
                      .replace("@@SHOWLINE@@", showline)
                      .replace("@@EXTRA@@", extra)
                      .replace("@@LANG@@", lang)
                      .replace("@@EMOJIS@@", " ".join(emojis) if emojis else "(none, use an empty string)")
                      .replace("@@CHARACTERS@@", ", ".join(profile["characters"]) or "(none)")
                      .replace("@@TEXT@@", text[:3500]))
        data = {}
        try:
            data = ask_json(llm, prompt)
            if not isinstance(data, dict):
                data = {}
        except Exception as ex:  # keep going with first-pass values
            print(f"      (metadata step failed for one clip: {ex})")
        c["title_original"] = c["title"]
        hook = pick_title(_title_list(data), text, None, bad)
        if hook is None:  # all options too short, copied or not allowed: ask once more
            try:
                d2 = ask_json(llm, prompt + RETRY_NOTE)
                if isinstance(d2, dict):
                    h2 = pick_title(_title_list(d2), text, None, bad)
                    if h2:
                        data, hook = d2, h2
            except Exception:
                pass
        orig = c["title"] if c["title"] and not bad(c["title"]) else ""
        c["title"] = (hook or orig or profile.get("fallback_title")
                      or f"Wild {profile['show'] or 'Game'} Moment")
        emoji = data.get("emoji")
        c["emoji"] = emoji if emoji in emojis else (emojis[0] if emojis else "")
        chars = []
        raw_chars = data.get("characters")
        for n in (raw_chars if isinstance(raw_chars, list) else []):
            real = known.get(norm(n))
            if real and real not in chars:
                chars.append(real)
        if name and name in known.values() and name not in chars:
            chars.insert(0, name)
        c["characters"] = chars
        c["summary"] = clean_text(data.get("summary", ""))
        c["question"] = clean_text(data.get("question", ""))
        if bad(c["summary"]):
            c["summary"] = ""
        if bad(c["question"]):
            c["question"] = ""
        raw_tags = data.get("tags")
        c["llm_tags"] = [clean_text(t) for t in (raw_tags if isinstance(raw_tags, list) else []) if isinstance(t, str)]
        cap = clean_text(data.get("caption", ""))
        cap = re.sub(r"[\"“”]", "", cap)
        if not (2 <= len(cap.split()) <= 8) or bad(cap):
            cap = ""
        c["overlay"] = with_name(cap or c["title"], name) if name else (cap or c["title"])
        c["yt_title"] = build_yt_title(c["title"], c["emoji"], name or profile["show"])
        c["description"] = build_description(c, profile)
        c["tags"] = build_tags(c, profile)
        c["tiktok_caption"] = build_tiktok(c, profile)
        c["yt_hashtags"] = hashtags(profile, "youtube")
        c["tt_hashtags"] = hashtags(profile, "tiktok")


def tighten(chosen, words, lead=0.15, tail=0.4):
    """Cut dead air: start just before the first spoken word, end just after the last."""
    for c in chosen:
        ws = [w for w in words if w["start"] >= c["start"] - 0.05 and w["end"] <= c["end"] + 0.3]
        if not ws:
            continue
        c["start"] = max(ws[0]["start"] - lead, 0.0)
        c["end"] = max(ws[-1]["end"] + tail, c["start"] + 1.0)


# ---------- 2c. Gameplay mode: find action moments without relying on speech ----------
def _robust_z(x, floor=1.0):
    """How unusual each value is vs the typical level. `floor` ignores variations too small to matter."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    med = float(np.median(x))
    mad = max(float(np.median(np.abs(x - med))) * 1.4826, floor)
    return (x - med) / mad


def audio_levels(video, hop=0.5):
    """Loudness (RMS) of the game audio every `hop` seconds. Empty array if there is no audio."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return np.zeros(0, np.float32)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(8000 * hop)
    m = len(x) // n
    if m == 0:
        return np.zeros(0, np.float32)
    return np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)


def motion_levels(video, hop=0.5):
    """How much the picture changes between samples (fast action, explosions, camera whips)."""
    w, h = 96, 54
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-an",
           "-vf", f"fps={1 / hop},scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    out, prev = [], None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    except OSError:
        return np.zeros(0, np.float32)
    while True:
        buf = proc.stdout.read(w * h)
        if len(buf) < w * h:
            break
        f = np.frombuffer(buf, np.uint8).astype(np.float32)
        out.append(0.0 if prev is None else float(np.abs(f - prev).mean()))
        prev = f
    proc.wait()
    return np.array(out, np.float32)


def action_scores(video, hop=0.5):
    print("      Scanning audio and motion for action moments (this can take a few minutes)...")
    a, m = audio_levels(video, hop), motion_levels(video, hop)
    n = max(len(a), len(m))
    if n == 0:
        return np.zeros(0, np.float32)

    def fit(x):
        out = np.zeros(n, np.float32)
        out[: len(x)] = x
        return out

    a, m = fit(a), fit(m)
    db = 20 * np.log10(a + 1e-5)
    prev = np.array([db[max(0, i - 4) : i].mean() if i > 0 else db[0] for i in range(n)])
    jump = np.maximum(0, db - prev)            # sudden loudness rise: crash, explosion, gunshot
    score = (0.35 * np.clip(_robust_z(db, 2.0), 0, 8)       # loudness (dB)
             + 0.30 * np.clip(_robust_z(jump, 2.0), 0, 8)   # sudden rise in loudness (dB)
             + 0.35 * np.clip(_robust_z(m, 1.0), 0, 8))     # picture change (grey levels)
    return np.convolve(score, np.ones(3) / 3, mode="same")


def _to_sec(s):
    secs = 0.0
    for p in s.strip().split(":"):
        secs = secs * 60 + float(p)
    return secs


def parse_skip(spec):
    """'2:10-5:40,12:00-14:30' -> [(130.0, 340.0), (720.0, 870.0)]"""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            a, b = part.split("-")
            out.append((_to_sec(a), _to_sec(b)))
        except ValueError:
            sys.exit(f"Bad --skip range '{part}'. Use start-end like 2:10-5:40 (several allowed, comma separated).")
    return out


def pick_action_clips(video, n, mn, mx, skip=(), hop=0.5):
    score = action_scores(video, hop)
    total = len(score) * hop
    if total < mn:
        sys.exit("The video is shorter than --min, so no clip can be made.")
    L = float(min(max((mn + mx) / 2, mn), mx))          # clip length, centre of your range
    s = score.astype(np.float64).copy()
    for a, b in skip:                                    # keep story/cutscene ranges out entirely
        lo, hi = max(0, int((a - 0.4 * L) / hop)), min(len(s), int((b + 0.6 * L) / hop) + 1)
        s[lo:hi] = -np.inf
    sep = int(L / hop)
    clips = []
    while len(clips) < n:
        i = int(np.argmax(s))
        if not np.isfinite(s[i]) or s[i] <= 1.0:        # nothing exciting left
            break
        peak = i * hop
        start = min(max(peak - 0.6 * L, 0.0), max(total - L, 0.0))
        end = min(start + L, total)
        z = float(s[i])
        s[max(0, i - sep) : i + sep + 1] = -np.inf
        if end - start < mn - 0.5:
            continue
        clips.append({"start": round(start, 2), "end": round(end, 2), "peak": round(peak, 2),
                      "score": int(min(99, max(1, 35 + 15 * z))), "title": "",
                      "reason": f"action peak at {mmss(peak)}"})
    clips.sort(key=lambda c: c["start"])
    if len(clips) < n:
        print(f"      Only {len(clips)} clear action moments found (wanted {n}).")
    return clips


GAME_FRAME_PROMPT = ("This is a frame from the video game @@GAME@@. In ONE short sentence, say what is "
                     "happening (vehicles, police, explosions, shooting, characters, setting). Describe only "
                     "what you can see. Do not guess names or story events.")


def describe_clip(video, start, end, model, game):
    """Short text notes of what a clip shows, from a local vision model, so titles stay truthful."""
    cap = cv2.VideoCapture(str(video))
    notes = []
    for label, frac in (("start", 0.1), ("middle", 0.5), ("end", 0.9)):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + (end - start) * frac) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        if frame.shape[1] > 960:
            frame = cv2.resize(frame, None, fx=960 / frame.shape[1], fy=960 / frame.shape[1])
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        try:
            r = requests.post(
                OLLAMA_URL,
                json={"model": model, "stream": False,
                      "messages": [{"role": "user", "content": GAME_FRAME_PROMPT.replace("@@GAME@@", game),
                                    "images": [base64.b64encode(buf.tobytes()).decode()]}],
                      "options": {"temperature": 0.2}},
                timeout=300,
            )
            r.raise_for_status()
            txt = clean_text(r.json()["message"]["content"])
            if txt:
                notes.append(f"{label}: {txt}")
        except Exception:
            continue
    cap.release()
    return " | ".join(notes)


META_PROMPT_GAME = """You write YouTube Shorts metadata for a short GAMEPLAY clip from @@SHOW@@.
You are NOT shown the video. Use ONLY the information at the bottom. Do not invent story events, character names, locations, mission names, numbers or outcomes that are not stated there.
If the information is thin, keep titles about the TYPE of moment (chase, crash, explosion, shootout, stunt, glitch, close call, funny physics) without specifics.
Never reveal story spoilers (endings, plot twists, cutscene dialogue).
Respond ONLY with JSON in exactly this shape:
{"titles":["...","...","..."],"caption":"...","emoji":"...","characters":["..."],"summary":"...","question":"...","tags":["..."]}

Rules:
- caption: big on-screen text for the first 2 seconds, 3 to 7 words, a reaction line about what is happening (example style: "Kai Cenat can't believe this"). It must be true to the clip, positive or neutral, in @@LANG@@.
- titles: 3 different hook titles, 4 to 9 words each, written in @@LANG@@. Create curiosity or tension so people must watch to the end. Put the key hook words in the FIRST 40 characters. Do not reveal the punchline. Must be true to the information below. No emojis, hashtags or quotation marks.
- emoji: the single best fit from this list: @@EMOJIS@@
- characters: only names from this list that are explicitly mentioned in the information below: @@CHARACTERS@@. Use [] otherwise.
- summary: one sentence (max 25 words) describing only what the information says happens.
- question: one short question for viewers that invites comments about this clip.
- tags: 6 to 10 search phrases (1 to 3 words each) viewers might type to find this clip.

WHAT HAPPENS (creator note and frame descriptions):
@@NOTES@@

COMMENTARY TRANSCRIPT (may be empty):
@@TEXT@@
"""


# ---------- 3. Reframe to 9:16 ----------
def video_size(video):
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def face_center_x(video, start, end, width):
    """Median x of the largest detected face across sampled frames."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():  # face data file missing: centre crop instead of crashing
        print("      (face detector data not found, using a centred crop)")
        return width / 2
    cap = cv2.VideoCapture(str(video))
    xs, n = [], 12
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            xs.append(x + w / 2)
    cap.release()
    if not xs:
        return width / 2
    xs.sort()
    return xs[len(xs) // 2]


# ---------- 4. Captions + render ----------
def fmt_time(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


COLORS = {
    "yellow": "00FFFF", "white": "FFFFFF", "green": "00FF00", "cyan": "FFFF00",
    "pink": "B469FF", "orange": "00A5FF", "red": "0000FF", "magenta": "FF00FF",
}

# Caption looks. pop = big bold text, current word pops in with a highlight colour.
CAPTION_STYLES = {
    "pop":     dict(font="Arial Black", size=92, outline=9, shadow=4, spacing=1,
                    upper=True, group=3, pop=True, highlight=True),
    "classic": dict(font="Arial", size=84, outline=6, shadow=2, spacing=0,
                    upper=True, group=3, pop=False, highlight=True),
    "clean":   dict(font="Arial", size=74, outline=5, shadow=2, spacing=0,
                    upper=False, group=5, pop=False, highlight=False),
}

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,@@FONT@@,@@SIZE@@,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,1,0,0,0,100,100,@@SPACING@@,0,1,@@OUTLINE@@,@@SHADOW@@,2,60,60,@@MARGINV@@,1
Style: Hook,Arial,112,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,9,4,8,60,60,@@HOOKV@@,1
Style: Title,Arial,70,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,3,16,0,8,80,80,230,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def parse_color(value):
    """Colour name (yellow, green...) or hex like #FFD400 -> ASS BBGGRR string."""
    v = value.strip().lower()
    if v in COLORS:
        return COLORS[v]
    m = re.fullmatch(r"#?([0-9a-f]{6})", v)
    if m:
        h = m.group(1)
        return (h[4:6] + h[2:4] + h[0:2]).upper()
    sys.exit(f"Unknown caption colour '{value}'. Use one of {', '.join(COLORS)} or a hex like #FFD400.")


def build_ass(words, clip_start, clip_end, path, captions=True, title=None, title_secs=0,
              style="pop", color="yellow", group=None, font=None, size=None, margin=520, hook=False):
    """Title at the top (optional) + styled captions (optional)."""
    st = CAPTION_STYLES[style]
    hl = parse_color(color)
    group = group or st["group"]
    header = (ASS_HEADER
              .replace("@@FONT@@", font or st["font"])
              .replace("@@SIZE@@", str(size or st["size"]))
              .replace("@@SPACING@@", str(st["spacing"]))
              .replace("@@OUTLINE@@", str(st["outline"]))
              .replace("@@SHADOW@@", str(st["shadow"]))
              .replace("@@MARGINV@@", str(margin))
              .replace("@@HOOKV@@", "620" if margin >= 800 else "150"))
    lines = [header]
    if title:
        clean = re.sub(r"[{}\\]", "", title).strip()
        wrapped = "\\N".join(textwrap.wrap(clean, 16 if hook else 22)[:3])
        t_end = (clip_end - clip_start) if title_secs <= 0 else min(title_secs, clip_end - clip_start)
        lines.append(f"Dialogue: 1,{fmt_time(0)},{fmt_time(t_end)},{'Hook' if hook else 'Title'},,0,0,0,,{{\\fad(0,200)}}{wrapped}\n")
    ws = [w for w in words if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.3
          and not has_cjk(w["w"])]
    if not captions:
        ws = []

    def fix(word):
        word = re.sub(r"[{}\\]", "", word)
        return word.upper() if st["upper"] else word

    for gi in range(0, len(ws), group):
        g = ws[gi : gi + group]
        if not st["highlight"]:
            # one event per group, no highlight (sentence-style captions)
            s = g[0]["start"] - clip_start
            e = max(g[-1]["end"] - clip_start, s + 0.2)
            text = " ".join(fix(x["w"]) for x in g)
            lines.append(f"Dialogue: 0,{fmt_time(s)},{fmt_time(e)},Default,,0,0,0,,{text}\n")
            continue
        for i, w in enumerate(g):
            s = w["start"] - clip_start
            e = (g[i + 1]["start"] if i + 1 < len(g) else w["end"]) - clip_start
            e = max(e, s + 0.08)
            if st["pop"]:
                tag = "{\\c&H" + hl + "&\\fscx80\\fscy80\\t(0,100,\\fscx118\\fscy118)}"
            else:
                tag = "{\\c&H" + hl + "&}"
            parts = []
            for j, x in enumerate(g):
                word = fix(x["w"])
                parts.append(tag + word + "{\\r}" if j == i else word)
            lines.append(
                f"Dialogue: 0,{fmt_time(s)},{fmt_time(e)},Default,,0,0,0,,{' '.join(parts)}\n"
            )
    Path(path).write_text("".join(lines), encoding="utf-8")


CAM_H, GAME_Y = 960, 1136   # stack layout: facecam fills the top half, gameplay sits in the bottom half


def render(video, start, end, ass_name, out_name, outdir, vw, vh, layout="crop", cam=None):
    # ass_name=None means no captions/title. layout: crop (follow the face) or blur (full frame on blurred bg)
    video = Path(video).resolve()
    landscape = vw / vh > 9 / 16 + 0.01
    if layout in ("blur", "stack") and landscape and (layout == "blur" or cam):
        bg = ("scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=10:3,"
              "scale=1080:1920,eq=brightness=-0.1")
        if layout == "stack":
            cx, cy, cw, ch = cam
            graph = (f"[0:v]split=3[a][b][c];[a]{bg}[bg];"
                     "[b]scale=1080:-2,setsar=1[g];"
                     f"[c]crop={cw}:{ch}:{cx}:{cy},scale=1080:{CAM_H}:force_original_aspect_ratio=increase,"
                     f"crop=1080:{CAM_H},setsar=1[f];"
                     f"[bg][g]overlay=0:{GAME_Y}[t1];[t1][f]overlay=0:0[v0]")
        else:
            graph = ("[0:v]split[a][b];"
                     f"[a]{bg}[bg];"
                     "[b]scale=1080:-2,setsar=1[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
        label = "v0"
        if ass_name:
            graph += f";[v0]subtitles={ass_name}[v]"
            label = "v"
        vf = ["-filter_complex", graph, "-map", f"[{label}]", "-map", "0:a?"]
    else:
        if landscape:  # crop around the face
            cw = int(vh * 9 / 16) // 2 * 2
            cx = face_center_x(video, start, end, vw)
            x = int(min(max(cx - cw / 2, 0), vw - cw)) // 2 * 2
            geometry = f"crop={cw}:{vh}:{x}:0,scale=1080:1920"
        else:  # already vertical: fit and pad
            geometry = (
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
            )
        vf = ["-vf", geometry + (f",subtitles={ass_name}" if ass_name else "")]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(video),
        *vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        out_name,
    ]
    # cwd=outdir so the subtitles filter gets a plain relative filename (avoids Windows path escaping)
    subprocess.run(cmd, cwd=outdir, check=True)


# ---------- 5. Thumbnails from the best frame of each clip ----------
FONT_CANDIDATES = ["impact.ttf", "ariblk.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]

VISION_PROMPT = (
    "Rate this video frame as a YouTube Shorts thumbnail from 1 to 10. "
    "High score: a clear, large face with a strong funny or surprising expression "
    "(eyes open, mouth open or smiling), sharp focus, good lighting. "
    "Low score: blurry or motion-blurred, eyes closed or half-blink, face turned away or tiny, "
    "dark, dull neutral expression, text cut across the frame. "
    'Respond ONLY with JSON: {"score": <1-10>}'
)


def need_pillow():
    try:
        import PIL  # noqa: F401
    except ImportError:
        sys.exit("Thumbnails need Pillow. Run:  pip install pillow   (with (venv) active), "
                 "or run again with --no-thumbs.")


def _cascades():
    base = cv2.data.haarcascades
    return (cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(base + "haarcascade_eye.xml"),
            cv2.CascadeClassifier(base + "haarcascade_smile.xml"))


def score_frame(frame, cascades):
    """Heuristic thumbnail score: big sharp face, eyes open, smile/open mouth. Returns (score, face box)."""
    face_c, eye_c, smile_c = cascades
    H, W = frame.shape[:2]
    scale = 640 / W if W > 640 else 1.0
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh = gray.shape[0]
    faces = [] if face_c.empty() else face_c.detectMultiScale(
        gray, 1.1, 6, minSize=(max(int(sh * 0.08), 24),) * 2)
    if len(faces) == 0:
        return min(cv2.Laplacian(gray, cv2.CV_64F).var() / 300, 1.0) * 10, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    roi = gray[y : y + h, x : x + w]
    score = 20 + min(h / (sh * 0.30), 1.0) * 30
    if not eye_c.empty():
        eyes = eye_c.detectMultiScale(roi[: int(h * 0.6)], 1.1, 5, minSize=(max(int(w * 0.12), 8),) * 2)
        score += 15 if len(eyes) >= 2 else (5 if len(eyes) == 1 else 0)
    if not smile_c.empty():
        smile = smile_c.detectMultiScale(roi[int(h * 0.55) :], 1.7, 20,
                                         minSize=(max(int(w * 0.25), 8), max(int(h * 0.08), 4)))
        score += 12 if len(smile) else 0
    score += min(cv2.Laplacian(roi, cv2.CV_64F).var() / 150, 1.0) * 15
    score += 5 if 70 <= roi.mean() <= 200 else 0
    return score, tuple(int(v / scale) for v in (x, y, w, h))


def vision_score(frame, model, prompt=None):
    """Ask a local vision model (via Ollama) to rate a frame 1-10. Returns None on failure."""
    f = frame
    if f.shape[1] > 960:
        f = cv2.resize(f, None, fx=960 / f.shape[1], fy=960 / f.shape[1])
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "stream": False, "format": "json",
                  "messages": [{"role": "user", "content": prompt or VISION_PROMPT,
                                "images": [base64.b64encode(buf.tobytes()).decode()]}],
                  "options": {"temperature": 0.1}},
            timeout=300,
        )
        r.raise_for_status()
        data = json.loads(r.json()["message"]["content"])
        return max(1.0, min(10.0, float(data["score"])))
    except Exception:
        return None


def best_frame(video, start, end, vision_model=None, keep=4):
    """Return (time_in_source, frame, face_box) of the best thumbnail candidate in [start, end]."""
    cap = cv2.VideoCapture(str(video))
    cascades = _cascades()
    span = max(end - start - 0.6, 0.1)
    step = max(0.4, span / 36)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    top, t = [], start + 0.3
    while t < end - 0.3:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:  # some files don't seek by time: try by frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
        if ok:
            score, box = score_frame(frame, cascades)
            top.append((score, t, frame, box))
            top.sort(key=lambda x: x[0], reverse=True)
            del top[keep:]
        t += step
    cap.release()
    if not top:  # OpenCV could not read frames: grab the middle frame with ffmpeg instead
        mid = (start + end) / 2
        tmp = Path(video).resolve().parent / "_thumb_tmp.png"
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{mid:.2f}", "-i", str(video),
                            "-frames:v", "1", str(tmp)], check=True)
            frame = cv2.imread(str(tmp))
        except Exception:
            frame = None
        finally:
            tmp.unlink(missing_ok=True)
        if frame is None:
            return None
        score, box = score_frame(frame, cascades)
        return mid, frame, box
    if vision_model:
        rated = []
        for score, tt, frame, box in top:
            v = vision_score(frame, vision_model)
            rated.append((score + (6 * v if v is not None else 0), score, tt, frame, box))
        rated.sort(key=lambda x: x[0], reverse=True)
        _, _, tt, frame, box = rated[0]
        return tt, frame, box
    _, tt, frame, box = top[0]
    return tt, frame, box


def _font(size, name=None):
    from PIL import ImageFont
    for f in ([name] if name else []) + FONT_CANDIDATES:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _wrap_px(words, font, draw, max_w):
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def make_thumbnail(frame, box, hook, out_text, out_clean, highlight_bgr="00FFFF", font_name=None,
                   zoom=None, max_size=160, force_place=None):
    """9:16 thumbnail: face-centred crop, punchy colour, big hook text that avoids covering the face."""
    from PIL import Image, ImageDraw, ImageEnhance
    H, W = frame.shape[:2]
    zoom = zoom or (1.25 if H >= 1000 else 1.0)
    ch = H / zoom
    cw = ch * 9 / 16
    if cw > W:
        cw, ch = W, W * 16 / 9
    fx, fy = (box[0] + box[2] / 2, box[1] + box[3] / 2) if box else (W / 2, H * 0.45)
    x0 = min(max(fx - cw / 2, 0), W - cw)
    y0 = min(max(fy - 0.55 * ch, 0), H - ch)
    crop = frame[int(y0) : int(y0 + ch), int(x0) : int(x0 + cw)]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((1080, 1920), Image.LANCZOS)
    img = ImageEnhance.Color(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    img.save(out_clean, quality=92)

    # lay the text out first (biggest size that fits in 3 lines), then decide top or bottom
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = re.sub(r"[{}\\]", "", hook).upper().split()
    size, lines, font = max_size, [], None
    while size >= 70:
        font = _font(size, font_name)
        lines = _wrap_px(words, font, measure, 930)
        if len(lines) <= 3:
            break
        size -= 6
    line_h = int(size * 1.08)
    block_h = line_h * len(lines)
    top_y = 140
    bottom_y = 1920 - 260 - block_h           # keep clear of the Shorts buttons
    place = "top"
    if box:
        s = 1920 / ch
        face_top, face_bottom = (box[1] - y0) * s, (box[1] + box[3] - y0) * s
        over_top = max(0, (top_y + block_h + 30) - face_top)         # text would cover the face
        over_bottom = max(0, face_bottom - (bottom_y - 30))
        place = "top" if over_top <= over_bottom else "bottom"
    place = force_place or place
    y = top_y if place == "top" else bottom_y

    # dark gradient on the text side so the words stay readable
    if place == "top":
        vals = [int(200 * max(0.0, 1 - r / (1920 * 0.42))) for r in range(1920)]
    else:
        vals = [int(200 * max(0.0, (r - 1920 * 0.52) / (1920 * 0.48))) for r in range(1920)]
    grad = Image.new("L", (1, 1920))
    grad.putdata(vals)
    img = Image.composite(Image.new("RGB", (1080, 1920), (0, 0, 0)), img, grad.resize((1080, 1920)))

    draw = ImageDraw.Draw(img)
    key = max((w for w in words if len(re.sub(r"\W", "", w)) >= 4), key=len, default=None)
    hl = (int(highlight_bgr[4:6], 16), int(highlight_bgr[2:4], 16), int(highlight_bgr[0:2], 16))
    stroke = max(6, size // 11)
    for line in lines:
        x = (1080 - draw.textlength(line, font=font)) / 2
        for w in line.split():
            color = hl if (key and w == key) else (255, 255, 255)
            draw.text((x, y), w, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += draw.textlength(w + " ", font=font)
        y += line_h
    img.save(out_text, quality=92)


VISION_PROMPT_GAME = (
    "Rate this video game frame as a YouTube Shorts thumbnail from 1 to 10. "
    "High score: dramatic action (explosion, crash, chase, police, big stunt, shootout), characters or "
    "vehicles clearly visible, vivid colours, sharp, high contrast. "
    "Low score: menu, map, loading or pause screen, blurry, very dark, empty road, text-heavy screens. "
    'Respond ONLY with JSON: {"score": <1-10>}'
)


def action_frame_score(frame, t, peak):
    """Thumbnail score for gameplay: sharp, colourful, contrasty, close to the action peak."""
    if frame.shape[1] > 640:
        frame = cv2.resize(frame, None, fx=640 / frame.shape[1], fy=640 / frame.shape[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    score = (min(cv2.Laplacian(gray, cv2.CV_64F).var() / 200, 1.0) * 35
             + min(float(hsv[..., 1].mean()) / 110, 1.0) * 25
             + min(float(gray.std()) / 60, 1.0) * 15
             + (max(0.0, 1 - abs(t - peak) / 4) * 25 if peak is not None else 12))
    return score * (0.5 if gray.mean() < 35 else 1.0)


def best_action_frame(video, start, end, peak=None, vision_model=None, keep=4):
    """Best thumbnail frame for gameplay (no faces needed). Returns (time, frame, None)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(0.4, max(end - start - 0.6, 0.1) / 36)
    top, t = [], start + 0.3
    while t < end - 0.3:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
        if ok:
            top.append((action_frame_score(frame, t, peak), t, frame))
            top.sort(key=lambda x: x[0], reverse=True)
            del top[keep:]
        t += step
    cap.release()
    if not top:
        return best_frame(video, start, end, None)       # ffmpeg fallback lives there
    if vision_model:
        rated = []
        for score, tt, frame in top:
            v = vision_score(frame, vision_model, VISION_PROMPT_GAME)
            rated.append((score + (6 * v if v is not None else 0), tt, frame))
        rated.sort(key=lambda x: x[0], reverse=True)
        return rated[0][1], rated[0][2], None
    return top[0][1], top[0][2], None


def make_thumbnail_blur(frame, hook, out_text, out_clean, highlight_bgr="00FFFF", font_name=None):
    """Gameplay thumbnail: full frame on a blurred background, big hook text above it."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    H, W = frame.shape[:2]
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    sc = max(1080 / W, 1920 / H)
    bg = rgb.resize((int(W * sc) + 1, int(H * sc) + 1), Image.BILINEAR)
    left, top = (bg.width - 1080) // 2, (bg.height - 1920) // 2
    bg = bg.crop((left, top, left + 1080, top + 1920)).filter(ImageFilter.GaussianBlur(40))
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    fh = int(round(H * 1080 / W))
    fg = rgb.resize((1080, fh), Image.LANCZOS)
    fg = ImageEnhance.Sharpness(ImageEnhance.Contrast(ImageEnhance.Color(fg).enhance(1.25)).enhance(1.1)).enhance(1.3)
    fg_top = (1920 - fh) // 2
    img = bg.copy()
    img.paste(fg, (0, fg_top))
    img.save(out_clean, quality=92)

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = re.sub(r"[{}\\]", "", hook).upper().split()
    zone_h = fg_top - 90
    size, lines, font = 150, [], None
    while size >= 60:
        font = _font(size, font_name)
        lines = _wrap_px(words, font, measure, 960)
        if len(lines) <= 3 and len(lines) * int(size * 1.08) <= zone_h:
            break
        size -= 6
    line_h = int(size * 1.08)
    y = max(60, (fg_top - line_h * len(lines)) // 2)
    draw = ImageDraw.Draw(img)
    key = max((w for w in words if len(re.sub(r"\W", "", w)) >= 4), key=len, default=None)
    hl = (int(highlight_bgr[4:6], 16), int(highlight_bgr[2:4], 16), int(highlight_bgr[0:2], 16))
    stroke = max(6, size // 11)
    for line in lines:
        x = (1080 - draw.textlength(line, font=font)) / 2
        for w in line.split():
            color = hl if (key and w == key) else (255, 255, 255)
            draw.text((x, y), w, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += draw.textlength(w + " ", font=font)
        y += line_h
    img.save(out_text, quality=92)


# ---------- 5b. Facecam stack layout, on-screen check, per-clip JSON ----------
def _cascade():
    c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return None if c.empty() else c


def video_duration(video):
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return n / fps if fps else 0.0


def corner_boxes(vw, vh):
    bw, bh = int(vw * 0.34) // 2 * 2, int(vh * 0.42) // 2 * 2
    return {"top-left": (0, 0, bw, bh), "top-right": (vw - bw, 0, bw, bh),
            "bottom-left": (0, vh - bh, bw, bh), "bottom-right": (vw - bw, vh - bh, bw, bh)}


def detect_facecam(video, vw, vh):
    """Find which corner holds the streamer's webcam by looking for a face there."""
    cascade = _cascade()
    if cascade is None:
        return None
    boxes = corner_boxes(vw, vh)
    cap = cv2.VideoCapture(str(video))
    total = video_duration(video)
    hits = {k: 0 for k in boxes}
    n = 14
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, total * (i + 0.5) / n * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in cascade.detectMultiScale(gray, 1.2, 5, minSize=(int(vh * 0.05),) * 2):
            fx, fy = x + w / 2, y + h / 2
            for k, (bx, by, bw, bh) in boxes.items():
                if bx <= fx < bx + bw and by <= fy < by + bh:
                    hits[k] += 1
    cap.release()
    best = max(hits, key=hits.get)
    if hits[best] < 3:
        return None
    print(f"      Facecam found in the {best} corner.")
    return boxes[best]


def parse_facecam(spec, video, vw, vh, outdir):
    """auto | top-left/top-right/bottom-left/bottom-right | x,y,w,h in pixels -> (x,y,w,h) or None."""
    spec = (spec or "auto").strip().lower()
    boxes = corner_boxes(vw, vh)
    if spec == "auto":
        cam = detect_facecam(video, vw, vh)
    elif spec in boxes:
        cam = boxes[spec]
    else:
        try:
            x, y, w, h = [int(float(v)) for v in spec.split(",")]
            assert w > 20 and h > 20 and x >= 0 and y >= 0 and x + w <= vw and y + h <= vh
            cam = (x // 2 * 2, y // 2 * 2, w // 2 * 2, h // 2 * 2)
        except (ValueError, AssertionError):
            sys.exit(f"Bad --facecam '{spec}'. Use auto, a corner (top-right...), or x,y,w,h in pixels "
                     f"inside the {vw}x{vh} frame.")
    if cam:
        try:  # a picture to check the crop box by eye
            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_MSEC, video_duration(video) / 2 * 1000)
            ok, frame = cap.read()
            cap.release()
            if ok:
                x, y, w, h = cam
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), max(3, vh // 200))
                cv2.imwrite(str(Path(outdir) / "facecam_check.jpg"), frame)
        except Exception:
            pass
    return cam


def compose_stack(frame, cam):
    """Same picture as the stack layout, for the thumbnail: blurred bg, gameplay band, facecam band."""
    H, W = frame.shape[:2]
    cw9 = min(W, int(H * 9 / 16))
    x0 = (W - cw9) // 2
    bg = cv2.GaussianBlur(cv2.resize(frame[:, x0 : x0 + cw9], (1080, 1920)), (0, 0), 25)
    canvas = (bg * 0.9).astype(np.uint8)
    gh = int(1080 * H / W)
    gh = min(gh, 1920 - GAME_Y)
    canvas[GAME_Y : GAME_Y + gh] = cv2.resize(frame, (1080, int(1080 * H / W)))[:gh]
    x, y, w, h = cam
    sc = max(1080 / w, CAM_H / h)
    big = cv2.resize(frame[y : y + h, x : x + w], (int(w * sc) + 1, int(h * sc) + 1))
    ox, oy = (big.shape[1] - 1080) // 2, (big.shape[0] - CAM_H) // 2
    canvas[0:CAM_H] = big[oy : oy + CAM_H, ox : ox + 1080]
    return canvas


def face_presence(video, start, end, region=None, n=8):
    """Share of sampled moments where a face is visible (in the facecam region if given). None if unknown."""
    cascade = _cascade()
    if cascade is None:
        return None
    cap = cv2.VideoCapture(str(video))
    seen = tot = 0
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + (end - start) * (i + 0.5) / n) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        if region:
            x, y, w, h = region
            frame = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tot += 1
        seen += 1 if len(cascade.detectMultiScale(gray, 1.2, 4, minSize=(40, 40))) else 0
    cap.release()
    return round(seen / tot, 2) if tot else None


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def write_clip_json(outdir, m, i, profile, source_name, titles):
    """clip_XX.json: copy-paste metadata for YouTube + TikTok, and self-checks against the campaign rules."""
    name, tag = profile["required_name"], (profile.get("required_hashtag") or profile.get("hashtag") or "")
    low = lambda t: str(t).lower()
    texts = [m["yt_title"], m["overlay"], m["description"], m["tiktok_caption"], " ".join(m["tags"])]
    checks = {
        "name_in_overlay": (not name) or (bool(m.get("overlay_on")) and low(name) in low(m["overlay"])),
        "name_in_youtube_title": (not name) or low(name) in low(m["yt_title"]),
        "name_in_tiktok_caption": (not name) or low(name) in low(m["tiktok_caption"]),
        "hashtag_in_youtube_description": (not tag) or low(tag) in low(m["description"]),
        "hashtag_in_tiktok_caption": (not tag) or low(tag) in low(m["tiktok_caption"]),
        "no_banned_words": not any(banned_hits(t, profile["banned_words"]) for t in texts),
        "no_japanese_text": not any(has_cjk(t) for t in texts) and not m.get("cjk_in_speech"),
        "tiktok_caption_max_150": len(m["tiktok_caption"]) <= 150,
        "title_unique": titles.count(m["yt_title"]) == 1,
        "thumbnail_made": bool(m.get("thumb")),
    }
    checks["all_pass"] = all(checks.values())
    vis = m.get("kai_visible")
    review = []
    if vis is not None and vis < 0.6:
        review.append(f"Face found in only {int(vis * 100)}% of sampled moments: check that {name or 'the streamer'} is on screen the whole clip.")
    if m.get("spoken_flags"):
        review.append("Spoken text contains words from your banned list: " + ", ".join(m["spoken_flags"]) + ". Watch the clip for negative framing.")
    review.append("Check by eye: no logos or watermarks in the picture, footage is from the allowed channels and after Nov 18, and the moment is positive or neutral.")
    data = {
        "file": m["file"],
        "thumbnail": m.get("thumb"),
        "duration_sec": m["duration"],
        "source": {"video": source_name, "start": hms(m["start"]), "end": hms(m["end"])},
        "overlay_text": m["overlay"],
        "publish_at_local": m.get("publish_at_local"),
        "publish_at_audience": m.get("publish_at"),
        "youtube": {
            "title": m["yt_title"],
            "description": m["description"],
            "tags": m["tags"],
            "hashtags": m["yt_hashtags"],
            "category": profile.get("category", "Entertainment"),
            "made_for_kids": False,
        },
        "tiktok": {
            "caption": m["tiktok_caption"],
            "caption_length": len(m["tiktok_caption"]),
            "hashtags": m["tt_hashtags"],
        },
        "checks": checks,
        "kai_visible_share": vis,
        "review_by_eye": review,
    }
    (Path(outdir) / f"clip_{i:02d}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return checks


# ---------- 6. Posting schedule + copy-paste upload plan ----------
def parse_times(csv):
    out = []
    for t in csv.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(dt.datetime.strptime(t, "%H:%M").time())
        except ValueError:
            sys.exit(f"Bad time '{t}' in --post-times. Use HH:MM, for example 09:00,14:00,20:00")
    if not out:
        sys.exit("--post-times needs at least one time.")
    return out


def make_slots(n, times, start_date):
    slots, day = [], start_date
    while len(slots) < n:
        slots += [dt.datetime.combine(day, t) for t in times]
        day += dt.timedelta(days=1)
    return slots[:n]


def mmss(sec):
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def write_plan(outdir, video_name, meta, times, start_date, shift=None, aud_label=""):
    """Slots are in the audience's time zone; `shift` converts them to your local time."""
    shift = shift or dt.timedelta(0)
    slots = make_slots(len(meta), times, start_date)
    order = sorted(range(len(meta)), key=lambda i: meta[i]["score"], reverse=True)  # best clips first
    for slot, i in zip(slots, order):
        meta[i]["publish_at"] = slot.strftime("%Y-%m-%d %H:%M")
        meta[i]["publish_at_local"] = (slot + shift).strftime("%Y-%m-%d %H:%M")
    ordered = sorted(meta, key=lambda m: m["publish_at"])
    when = ", ".join(t.strftime("%H:%M") for t in times)
    out = [
        f"# Upload plan: {video_name}\n",
        f"{len(meta)} Shorts, {len(times)} per day at {when} "
        + (f"({aud_label}); times in YOUR time are shown first. " if aud_label else "(your computer's local time). ")
        + "Your best-scoring clips go out first.\n",
        "Checklist: keep the same posting times every day for at least 2 weeks, stay on one kind of "
        "content, and in Studio use the thumbnail listed below.\n",
    ]
    for n, m in enumerate(ordered, 1):
        pub = f"- Publish (your time): **{m['publish_at_local']}**" + (f"  = {m['publish_at']} {aud_label}" if aud_label else "")
        out += [
            f"\n---\n\n## {n}. {m['yt_title']}\n",
            pub,
            f"- Video file: `{m['file']}` ({m['duration']:.0f}s), metadata: `{m['file'][:-4]}.json`",
            f"- Thumbnail: `{m.get('thumb', '(none)')}`"
            + (f" (best frame at {mmss(m['thumb_time'])} in the clip; "
               "or pick that moment in Studio's frame picker)" if m.get("thumb_time") is not None else ""),
            "\n**Title** (key words are in the first 40 characters)\n",
            f"```text\n{m['yt_title']}\n```\n",
            "**Description**\n",
            f"```text\n{m['description']}\n```\n",
            "**Tags** (Studio > Details > Show more > Tags)\n",
            f"```text\n{', '.join(m['tags'])}\n```\n",
            "**TikTok caption**\n",
            f"```text\n{m['tiktok_caption']}\n```",
        ]
    (outdir / "upload_plan.md").write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Local AI clipper for Shorts/Reels/TikTok")
    ap.add_argument("video")
    ap.add_argument("--clips", type=int, default=None, help="how many clips to make (default 5)")
    ap.add_argument("--min", dest="mn", type=int, default=None, help="min clip seconds (default 10)")
    ap.add_argument("--max", dest="mx", type=int, default=None, help="max clip seconds (default 20)")
    ap.add_argument("--llm", default="qwen2.5:14b", help="Ollama model name")
    ap.add_argument("--whisper", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    ap.add_argument("--compute-type", default="auto", help="auto, int8, float16...")
    ap.add_argument("--language", default=None, help="e.g. en, ne, hi (default: auto-detect)")
    ap.add_argument("--window", type=int, default=480, help="seconds of transcript per LLM call")
    ap.add_argument("--prefilter", type=int, default=None,
                    help="long videos: only send the N most hyped transcript windows to the LLM (0 = all)")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--no-captions", action="store_true", help="make clips without burned-in captions")
    ap.add_argument("--caption-style", default="pop", choices=list(CAPTION_STYLES),
                    help="pop (bold, animated), classic, or clean")
    ap.add_argument("--caption-color", default="yellow",
                    help="highlight colour: yellow, green, cyan, pink, orange, red, magenta, white or #RRGGBB")
    ap.add_argument("--caption-words", type=int, default=None, help="words shown at once")
    ap.add_argument("--caption-size", type=int, default=None, help="caption font size")
    ap.add_argument("--font", default=None, help="caption font name, e.g. 'Arial Black'")
    ap.add_argument("--title-lang", default="English",
                    help="language for the AI titles (default English)")
    ap.add_argument("--no-title", action="store_true", help="don't show the AI title at the top")
    ap.add_argument("--title-seconds", type=float, default=None,
                    help="how long the on-screen text shows (0 = whole clip)")
    ap.add_argument("--mode", default="speech", choices=["speech", "gameplay"],
                    help="speech = pick clips from what is said; gameplay = pick action moments (audio + motion)")
    ap.add_argument("--commentary", action="store_true", help="gameplay mode: also transcribe your voice (captions, titles)")
    ap.add_argument("--layout", default=None, choices=["crop", "blur", "stack"],
                    help="9:16 layout: crop (follow the face), blur (full frame on blurred background), "
                         "stack (gameplay on top, facecam below)")
    ap.add_argument("--facecam", default=None,
                    help="stack layout: auto, a corner (top-right, bottom-left...) or x,y,w,h in pixels")
    ap.add_argument("--clip-note", default="", help="gameplay: what is in the video, to keep titles truthful")
    ap.add_argument("--skip", default="", help="time ranges to exclude, e.g. 2:10-5:40,12:00-14:30 (cutscenes, breaks)")
    ap.add_argument("--vision", default=None,
                    help="Ollama vision model that describes gameplay frames for titles and rates thumbnails, e.g. qwen3-vl:2b")
    ap.add_argument("--profile", default=None, help="channel json with your show name, tags, hashtag, rules")
    ap.add_argument("--no-thumbs", action="store_true", help="don't make thumbnails")
    ap.add_argument("--thumb-vision", default=None,
                    help="optional Ollama vision model to pick the best thumbnail frame, e.g. qwen3-vl:2b")
    ap.add_argument("--thumb-font", default=None, help="thumbnail font file name, e.g. impact.ttf")
    ap.add_argument("--post-times", default=None,
                    help="daily upload times, HH:MM comma separated (one Short per time)")
    ap.add_argument("--start-date", default=None, help="first upload day YYYY-MM-DD (default: tomorrow)")
    ap.add_argument("--audience-offset", type=float, default=None,
                    help="UTC offset of your audience, e.g. -5 for US Eastern. Post times are in this zone")
    ap.add_argument("--local-offset", type=float, default=None,
                    help="your own UTC offset, e.g. 5.75 for Nepal (default: this computer's zone)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # emoji-safe on Windows consoles
        except Exception:
            pass

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"Video not found: {video}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Install it first (see README.md).")
    try:
        requests.get("http://localhost:11434", timeout=3)
    except requests.RequestException:
        sys.exit("Ollama is not running. Install it and run: ollama pull " + args.llm)

    profile = load_profile(args.profile)
    D = profile.get("defaults") or {}
    pick = lambda v, key, hard: v if v is not None else D.get(key, hard)
    args.clips = pick(args.clips, "clips", 5)
    args.mn = pick(args.mn, "min", 10)
    args.mx = pick(args.mx, "max", 20)
    args.language = pick(args.language, "language", None)
    args.post_times = pick(args.post_times, "post_times", "09:00,14:00,20:00")
    args.prefilter = pick(args.prefilter, "prefilter", 0)
    args.facecam = pick(args.facecam, "facecam", "auto")
    args.layout = pick(args.layout, "layout", None)
    args.title_seconds = pick(args.title_seconds, "title_seconds", 0)
    name = profile["required_name"]

    ml = profile["mandatory_line"].strip()
    if (not ml or ml.upper().startswith("PASTE")) and profile.get("mandatory_line_required", True):
        print("WARNING: no mandatory line set. Edit your channel json (mandatory_line) so the descriptions are complete.")
    if name and args.no_title:
        print(f"NOTE: ignoring --no-title: the on-screen text with '{name}' is a campaign rule.")
        args.no_title = False
    times = parse_times(args.post_times)
    aud = args.audience_offset if args.audience_offset is not None else profile.get("audience_utc_offset")
    loc = args.local_offset if args.local_offset is not None else D.get("local_utc_offset")
    if loc is None:
        loc = dt.datetime.now().astimezone().utcoffset().total_seconds() / 3600
    shift = dt.timedelta(hours=loc - aud) if aud is not None else None
    aud_label = f"UTC{aud:+g}" if aud is not None else ""
    try:
        if args.start_date:
            start_date = dt.date.fromisoformat(args.start_date)
        else:
            now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=aud if aud is not None else loc)
            start_date = now.date() + dt.timedelta(days=1)
    except ValueError:
        sys.exit("--start-date must look like 2026-11-20")
    if not args.no_thumbs:
        need_pillow()

    outdir = Path(args.out) / video.stem
    outdir.mkdir(parents=True, exist_ok=True)

    gameplay = args.mode == "gameplay"
    layout = args.layout or ("blur" if gameplay else "crop")
    thumb_vision = args.thumb_vision or args.vision
    skip = parse_skip(args.skip)
    vw, vh = video_size(video)
    dur_total = video_duration(video)
    if name and (vh > vw or 0 < dur_total <= 180):
        print("WARNING: this file looks like a Short or a very short video. The campaign needs your OWN cuts "
              "from the full stream/VOD, not re-posts of existing Shorts.")
    cam = None
    if layout == "stack":
        if vw / vh <= 9 / 16 + 0.01:
            print("      (video is already vertical: stack layout skipped)")
            layout = "crop"
        else:
            cam = parse_facecam(args.facecam, video, vw, vh, outdir)
            if cam is None:
                print("WARNING: no facecam found, so using the blur layout. Pass --facecam top-right (or x,y,w,h) "
                      "to set it by hand.")
                layout = "blur"
            else:
                print(f"      Facecam box x,y,w,h = {cam[0]},{cam[1]},{cam[2]},{cam[3]} "
                      f"(preview: {outdir / 'facecam_check.jpg'})")

    if gameplay and not args.commentary:
        segments = []
        print("[1/4] Gameplay mode: skipping speech transcription (add --commentary if you talk over the game).")
    else:
        segments = transcribe(video, args.whisper, args.device, args.compute_type, args.language)
        if not segments and not gameplay:
            sys.exit("No speech detected in the video.")
    if segments:
        (outdir / "transcript.json").write_text(json.dumps(segments, indent=2), encoding="utf-8")

    if gameplay:
        print("[2/4] Finding the biggest action moments...")
        chosen = pick_action_clips(video, args.clips, args.mn, args.mx, skip)
        if not chosen:
            sys.exit("No clear action moments found. Try a lower --min or a longer video.")
        if args.vision:
            print(f"      Describing each clip with '{args.vision}' so titles match what happens...")
            for c in chosen:
                c["notes"] = describe_clip(video, c["start"], c["end"], args.vision, profile["show"] or "a video game")
    else:
        chosen = pick_clips(segments, args.llm, args.clips, args.mn, args.mx, args.window,
                            profile.get("pick_guidance", ""), skip, args.prefilter, video)
        if not chosen:
            sys.exit("The LLM returned no usable clips. Try a bigger model or a lower --min.")

    all_words = [w for s in segments for w in s["words"]]
    if not gameplay:
        tighten(chosen, all_words)
    make_metadata(chosen, segments, args.llm, args.title_lang, profile, args.mode, args.clip_note)

    print(f"[3/4] + [4/4] Rendering {len(chosen)} clips" + ("" if args.no_thumbs else " and thumbnails") + "...")
    margin = 800 if layout == "stack" else 520
    meta = []
    for i, c in enumerate(chosen, 1):
        ass_name, out_name = f"clip_{i:02d}.ass", f"clip_{i:02d}.mp4"
        use_title = bool(c["title"]) and not args.no_title
        use_ass = use_title or not args.no_captions
        c["overlay_on"] = use_title
        if use_ass:
            build_ass(all_words, c["start"], c["end"], outdir / ass_name,
                      captions=not args.no_captions,
                      title=c["overlay"] if use_title else None,
                      title_secs=args.title_seconds, hook=bool(name),
                      style=args.caption_style, color=args.caption_color,
                      group=args.caption_words, font=args.font, size=args.caption_size, margin=margin)
        c["cjk_in_speech"] = any(has_cjk(w["w"]) for w in all_words
                                 if c["start"] - 0.05 <= w["start"] and w["end"] <= c["end"] + 0.3)
        render(video, c["start"], c["end"], ass_name if use_ass else None,
               out_name, outdir, vw, vh, layout, cam)
        c["kai_visible"] = face_presence(video, c["start"], c["end"], cam) if name else None
        if not args.no_thumbs:
            try:
                if gameplay:
                    bf = best_action_frame(video, c["start"], c["end"], c.get("peak"), thumb_vision)
                else:
                    bf = best_frame(video, c["start"], c["end"], thumb_vision)
                if bf:
                    tt, frame, box = bf
                    c["thumb"] = f"thumb_{i:02d}.jpg"
                    if layout == "stack" and cam:
                        make_thumbnail(compose_stack(frame, cam), None, c["title"], outdir / c["thumb"],
                                       outdir / f"thumb_{i:02d}_clean.jpg",
                                       parse_color(args.caption_color), args.thumb_font, zoom=1.0, max_size=120, force_place="bottom")
                    elif gameplay and layout == "blur" and vw / vh > 9 / 16 + 0.01:
                        make_thumbnail_blur(frame, c["title"], outdir / c["thumb"],
                                            outdir / f"thumb_{i:02d}_clean.jpg",
                                            parse_color(args.caption_color), args.thumb_font)
                    else:
                        make_thumbnail(frame, box, c["title"], outdir / c["thumb"],
                                       outdir / f"thumb_{i:02d}_clean.jpg",
                                       parse_color(args.caption_color), args.thumb_font)
                    c["thumb_time"] = round(tt - c["start"], 1)
                else:
                    print(f"      (no usable frame found for the thumbnail of clip {i})")
            except Exception as ex:
                print(f"      (thumbnail failed for clip {i}: {ex})")
        dur = c["end"] - c["start"]
        meta.append({**c, "duration": round(dur, 1), "file": out_name})
        print(f"      {out_name}  [{c['start']:.0f}s-{c['end']:.0f}s] {dur:.0f}s long, "
              f"score {c['score']}  {c['title']}")

    write_plan(outdir, video.stem, meta, times, start_date, shift, aud_label)
    titles = [m["yt_title"] for m in meta]
    failed = []
    for i, m in enumerate(meta, 1):
        checks = write_clip_json(outdir, m, i, profile, video.name, titles)
        m["checks_pass"] = checks["all_pass"]
        if not checks["all_pass"]:
            failed.append((m["file"], [k for k, v in checks.items() if v is False and k != "all_pass"]))
    (outdir / "clips.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Everything (clips, thumbnails, clip_XX.json, upload_plan.md) is in: {outdir.resolve()}")
    if failed:
        print("Self-check failed for:")
        for f, ks in failed:
            print(f"  {f}: {', '.join(ks)}")
    else:
        print("Self-check: every clip passed the automatic rule checks (still review each clip by eye).")


if __name__ == "__main__":
    main()