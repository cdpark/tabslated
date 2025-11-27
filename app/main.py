# app.py — local build that worked before deployment
import os, subprocess, tempfile, pathlib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse

# ====== Whisper (self-hosted, free) ======
from faster_whisper import WhisperModel

WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")

# Lazy-load so the app can start even if the model fails; force CPU to avoid CUDA/cuDNN.
WHISPER = None
def get_whisper():
    global WHISPER
    if WHISPER is None:
        # int8 is fast on CPU; you can try "medium" later if your machine is beefy
        WHISPER = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return WHISPER


def _lrc_ts(t: float | None) -> str:
    if t is None: t = 0.0
    m = int(t // 60); s = t - 60*m
    return f"[{m:02d}:{s:05.2f}]"

def whisper_to_lrc(audio_path: str, per_word=False) -> str:
    """
    Transcribe 'audio_path' with faster-whisper and return LRC text.
    - per_word=False: one LRC line per ASR segment (good default)
    - per_word=True : one LRC line per word (karaoke-style; noisier on singing)
    """
    wm = get_whisper()
    segments, info = wm.transcribe(
        audio_path,
        beam_size = 5,
        vad_filter = True,
        word_timestamps = True
    )
    if per_word:
        lines = []
        for seg in segments:
            for w in (seg.words or []):
                txt = (w.word or "").strip()
                if txt:
                    lines.append(f"{_lrc_ts(w.start)} {txt}")
        return "\n".join(lines)
    else:
        lines = []
        for seg in segments:
            txt = (seg.text or "").strip()
            if txt:
                lines.append(f"{_lrc_ts(seg.start)} {txt}")
        return "\n".join(lines)

# ====== Robust chord post-processing & pattern smoothing ======
import re
from collections import Counter, defaultdict

# ---------- Key detection + diatonic correction ----------
try:
    import librosa
except Exception:
    librosa = None

PC_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
NOTE_TO_PC = {
    "C":0,"B#":0,
    "C#":1,"Db":1,
    "D":2,
    "D#":3,"Eb":3,
    "E":4,"Fb":4,
    "F":5,"E#":5,
    "F#":6,"Gb":6,
    "G":7,
    "G#":8,"Ab":8,
    "A":9,
    "A#":10,"Bb":10,
    "B":11,"Cb":11,
}

# Krumhansl key profiles (normalized)
_KR_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=float)
_KR_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=float)
_KR_MAJ /= np.linalg.norm(_KR_MAJ); _KR_MIN /= np.linalg.norm(_KR_MIN)

def estimate_key_krumhansl(wav_path: str, sr=22050):
    """Estimate global key by correlating mean chroma with Krumhansl profiles."""
    if librosa is None:
        return None, None
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)   # (12, T)
    prof = chroma.mean(axis=1) + 1e-9
    prof /= np.linalg.norm(prof)
    best = (-1e9, 0, 'maj')
    for tonic in range(12):
        rot = np.roll(prof, -tonic)
        cmaj = float(np.dot(rot, _KR_MAJ))
        cmin = float(np.dot(rot, _KR_MIN))
        if cmaj > best[0]: best = (cmaj, tonic, 'maj')
        if cmin > best[0]: best = (cmin, tonic, 'min')
    return best[1], best[2]

def diatonic_quality_for(root_pc: int, tonic_pc: int, mode: str):
    """Return 'maj'|'min'|'dim' or None if root not diatonic in that mode."""
    deg = (root_pc - tonic_pc) % 12
    if mode == 'maj':
        mapping = {0:'maj', 2:'min', 4:'min', 5:'maj', 7:'maj', 9:'min', 11:'dim'}
        return mapping.get(deg)
    else:  # natural minor
        mapping = {0:'min', 2:'dim', 3:'maj', 5:'min', 7:'min', 8:'maj', 10:'maj'}
        return mapping.get(deg)

_CHORD_RE = re.compile(r'^([A-G][b#]?)(m|dim|aug)?$')

def key_aware_correct(segments, tonic_pc, mode,
                      drop_out_of_scale_if_short=True,
                      short_thresh=1.2):
    """Snap each chord's quality to the diatonic quality for the detected key."""
    if tonic_pc is None or mode not in ('maj','min'):
        return segments
    out = []
    for s in segments:
        name = s["chord"]
        if name == "N.C.":
            out.append(s); continue
        m = _CHORD_RE.match(name)
        if not m:
            out.append(s); continue
        root, _qual = m.group(1), (m.group(2) or '')
        root_pc = NOTE_TO_PC.get(root)
        if root_pc is None:
            out.append(s); continue
        target = diatonic_quality_for(root_pc, tonic_pc, mode)
        if target:
            new = root + ('m' if target=='min' else ('dim' if target=='dim' else ''))
            out.append({**s, "chord": new})
        else:
            if drop_out_of_scale_if_short and (s["end"]-s["start"]) < short_thresh:
                out.append({**s, "chord": "N.C."})
            else:
                out.append(s)
    return _merge_adjacent(out)

# ---- TUNABLES ----
HOP              = 0.50     # seconds per token for rasterization
MODE_WIN         = 5        # odd number; token mode filter window
MIN_CHORD_DUR    = 1.20
MIN_NC_DUR       = 0.60
MIN_DIM_DUR      = 2.00
K_CANDIDATES     = [3,4,5,6,7,8]
MIN_OCC          = 2
MIN_SEP_TOK      = 8
MAX_NC_RATIO     = 0.50
ALT_THRESH       = 0.35
STRICT_ENFORCE   = True
KEEP_OUTSIDE     = True

# ---- Label simplification ----
_TRIAD_RE = re.compile(r'^([A-G][b#]?)(?::([A-Za-z0-9+\-]+))?$')

def simplify_label(raw: str) -> str:
    lab = raw.strip().strip('"').strip()
    if lab == 'N':
        return 'N.C.'
    m = _TRIAD_RE.match(lab)
    if not m:
        if lab.endswith('m'):
            return lab  # e.g., "Cm"
        return lab
    root = m.group(1)
    qual = (m.group(2) or 'maj').lower()
    if qual.startswith('min') or qual.startswith('m'):
        return root + 'm'
    if 'dim' in qual or '°' in qual or 'o' in qual:
        return root + 'dim'
    if 'aug' in qual or '+' in qual:
        return root + 'aug'
    return root

def collapse_colors(name: str) -> str:
    if name == 'N.C.': return name
    if name.endswith('m'): return name
    if name.endswith('dim') or name.endswith('aug'): return name
    base = re.match(r'^([A-G][b#]?)(?:.*)?$', name)
    return base.group(1) if base else name

# ---- Parse LAB (2-col or 3-col), then simplify ----
_LAB3 = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$')
_LAB2 = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)\s+(.+?)\s*$')

def parse_lab(lab_text: str):
    segs = []
    for line in (lab_text or "").splitlines():
        m = _LAB3.match(line)
        if m:
            s = float(m.group(1)); e = float(m.group(2)); raw = m.group(3).split()[0]
            simp = collapse_colors(simplify_label(raw))
            segs.append({"start": round(s,3), "end": round(e,3), "chord": simp})
    if segs:
        return segs

    events = []
    for line in (lab_text or "").splitlines():
        m = _LAB2.match(line)
        if not m:
            continue
        t = float(m.group(1)); raw = m.group(2).split()[0]
        simp = collapse_colors(simplify_label(raw))
        events.append((t, simp))
    if len(events) < 2:
        return []
    out = []
    for i in range(len(events)-1):
        t0, lab0 = events[i]; t1, _ = events[i+1]
        if lab0 == 'N.C.':
            continue
        out.append({"start": round(t0,3), "end": round(t1,3), "chord": lab0})
    return out

def _simplify(raw: str) -> str:
    lab = raw.strip().strip('"').strip()
    if lab == 'N': return 'N.C.'
    m = _TRIAD_RE.match(lab)
    if m:
        root = m.group(1)
        qual = (m.group(2) or 'maj').lower()
        if qual.startswith('min') or qual.startswith('m'): return root + 'm'
        if 'dim' in qual or '°' in qual or 'o' in qual:   return root + 'dim'
        if 'aug' in qual or '+' in qual:                   return root + 'aug'
        return root
    if lab.endswith('m'): return lab
    m = re.match(r'^([A-G][b#]?)', lab)
    return m.group(1) if m else lab

def _merge_adjacent(segs):
    if not segs: return []
    out = [segs[0].copy()]
    for s in segs[1:]:
        if s["chord"] == out[-1]["chord"] and abs(s["start"] - out[-1]["end"]) < 1e-3:
            out[-1]["end"] = s["end"]
        else:
            out.append(s.copy())
    return out

def _simplify_segments(segs):
    out = []
    for s in segs:
        out.append({"start": s["start"], "end": s["end"], "chord": _simplify(s["chord"])})
    return _merge_adjacent(out)

def _prune_blips(segs):
    if not segs: return []
    S = _merge_adjacent(segs)

    # bridge tiny N.C. between identical neighbors
    i = 1
    while i < len(S)-1:
        a, b, c = S[i-1], S[i], S[i+1]
        if b["chord"] == "N.C." and (b["end"]-b["start"]) < MIN_NC_DUR and a["chord"] == c["chord"]:
            a["end"] = c["start"]
            S.pop(i); S = _merge_adjacent(S); continue
        i += 1

    # short dim/aug → N.C.
    for s in S:
        if (s["chord"].endswith("dim") or s["chord"].endswith("aug")) and (s["end"]-s["start"]) < MIN_DIM_DUR:
            s["chord"] = "N.C."
    S = _merge_adjacent(S)

    # short non-N.C. → absorb into longer neighbor
    out = []
    for idx, s in enumerate(S):
        d = s["end"] - s["start"]
        if s["chord"] != "N.C." and d < MIN_CHORD_DUR:
            prev = out[-1] if out else None
            nxt  = S[idx+1] if idx+1 < len(S) else None
            if prev and nxt and prev["chord"] == nxt["chord"]:
                s["chord"] = prev["chord"]
            elif prev and (not nxt or (prev["end"]-prev["start"]) >= (nxt["end"]-nxt["start"])):  # noqa
                s["chord"] = prev["chord"]
            elif nxt:
                s["chord"] = nxt["chord"]
            else:
                s["chord"] = "N.C."
        out.append(s)
    return _merge_adjacent(out)

# ----- tokenization -----
def _rasterize(segs, hop=HOP):
    T = segs[-1]["end"] if segs else 0.0
    if T <= 0: return [], np.array([])
    times = np.arange(0.0, T, hop)
    toks = []
    j = 0
    for t in times:
        while j < len(segs) and segs[j]["end"] <= t: j += 1
        if j < len(segs) and segs[j]["start"] <= t < segs[j]["end"]:
            toks.append(segs[j]["chord"])
        else:
            toks.append("N.C.")
    return toks, times

def _mode_filter(tokens, win=MODE_WIN):
    if win < 3 or win % 2 == 0: return tokens[:]
    out = tokens[:]; half = win//2
    for i in range(len(tokens)):
        a = max(0, i-half); b = min(len(tokens), i+half+1)
        sub = tokens[a:b]
        counts = Counter(sub)
        best = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]!="N.C."), reverse=True)[0][0]
        out[i] = best
    return out

def _derasterize(tokens, times):
    if not tokens: return []
    segs = []
    cur = tokens[0]; start = float(times[0])
    for i in range(1, len(tokens)):
        if tokens[i] != cur:
            segs.append({"start": round(start,3), "end": round(float(times[i]),3), "chord": cur})
            cur = tokens[i]; start = float(times[i])
    tail = float(times[-1] + (times[1]-times[0]) if len(times)>1 else start+HOP)
    segs.append({"start": round(start,3), "end": round(tail,3), "chord": cur})
    segs = [s for s in segs if s["chord"]!="N.C." or (s["end"]-s["start"]) >= MIN_NC_DUR]
    return _merge_adjacent(segs)

# ---- Repeated-section detection ----
def _nonoverlapping_starts(all_starts, min_sep):
    sel = []; last = -10**9
    for s in sorted(all_starts):
        if s - last >= min_sep:
            sel.append(s); last = s
    return sel

def _score_window(window, starts):
    nc_ratio = window.count("N.C.")/len(window)
    unique = len({x for x in window if x!="N.C."})
    return (len(starts) * len(window) * max(0.0, 1.0 - nc_ratio)) + unique

def find_best_pattern(tokens, k_list=K_CANDIDATES, min_occ=MIN_OCC, min_sep=MIN_SEP_TOK):
    best = (None, [])
    best_score = -1
    for k in k_list:
        if len(tokens) < k: continue
        table = defaultdict(list)
        for i in range(len(tokens)-k+1):
            win = tuple(tokens[i:i+k])
            if win.count("N.C.")/k > MAX_NC_RATIO:  # skip
                continue
            if len({x for x in win if x!="N.C."}) < 2:
                continue
            table[win].append(i)
        for win, idxs in table.items():
            starts = _nonoverlapping_starts(idxs, min_sep)
            if len(starts) < min_occ:
                continue
            sc = _score_window(list(win), starts)
            if sc > best_score:
                best_score = sc
                best = (list(win), starts)
    return best

def consensus_for_pattern(tokens, starts, k):
    cons = []; alts = []
    for off in range(k):
        votes = [tokens[s+off] for s in starts if s+off < len(tokens)]
        votes = [v for v in votes if v!="N.C."]
        if votes:
            c = Counter(votes)
            top = c.most_common(1)[0][0]
            cons.append(top)
            allowed = {lab for lab, n in c.items() if n/sum(c.values()) >= ALT_THRESH}
            alts.append(allowed)
        else:
            cons.append("N.C."); alts.append(set())
    return cons, alts

def enforce_pattern_variable(tokens, times, pattern, starts, strict=STRICT_ENFORCE):
    k = len(pattern)
    cons, alts = consensus_for_pattern(tokens, starts, k)
    toks = tokens[:]
    covered = np.zeros(len(tokens), dtype=bool)
    for s in starts:
        for off in range(k):
            i = s+off
            if i >= len(toks): break
            covered[i] = True
            want = cons[off]
            allowed = alts[off] | {want}
            if toks[i] in allowed:
                continue
            if strict or toks[i]=="N.C.":
                toks[i] = want
    toks = _mode_filter(toks, win=MODE_WIN)
    return _derasterize(toks, times), covered

def postprocess_variable_pattern(raw_segments):
    simp = _simplify_segments(raw_segments)
    base = _prune_blips(simp)
    tokens, times = _rasterize(base)
    if not tokens:
        return base, [], []
    tokens = _mode_filter(tokens, win=MODE_WIN)
    pattern, starts = find_best_pattern(tokens)
    if not pattern:
        rough = _derasterize(tokens, times)
        return _prune_blips(rough), [], []
    enforced, _covered_mask = enforce_pattern_variable(tokens, times, pattern, starts, strict=STRICT_ENFORCE)
    final = _prune_blips(enforced)

    secs = []
    for s in starts:
        t0 = float(times[s]); t1 = float(times[min(s+len(pattern), len(times)-1)])
        secs.append({"start": round(t0,3), "end": round(t1,3)})
    secs.sort(key=lambda x: x["start"])
    merged = []
    for sec in secs:
        if not merged or sec["start"] > merged[-1]["end"] + 0.01:
            merged.append(sec)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], sec["end"])
    return final, pattern, merged

# ===== Lyrics parsing & chord-over-lyrics alignment =====
_LRC_TIME = re.compile(r"\[(\d{1,2}):(\d{2})(?:\.(\d{1,2}))?\]")

def parse_lrc(lrc_text: str):
    out = []
    for raw in (lrc_text or "").splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        times = list(_LRC_TIME.finditer(line))
        text = _LRC_TIME.sub("", line).strip()
        if not times:
            if text:
                out.append({"start": None, "end": None, "text": text})
            continue
        starts = []
        for m in times:
            mm = int(m.group(1))
            ss = int(m.group(2))
            frac = m.group(3)
            if frac is None:
                t = mm*60 + ss
            else:
                scale = 10 if len(frac) == 1 else 100
                t = mm*60 + ss + int(frac)/scale
            starts.append(float(t))
        if text:
            out.append({"start": min(starts), "end": None, "text": text})
    for i in range(len(out)-1):
        if out[i]["start"] is not None and out[i+1]["start"] is not None:
            out[i]["end"] = out[i+1]["start"]
    return out

def interpolate_words(line):
    words = line["text"].split()
    if not words:
        return []
    t0, t1 = line.get("start"), line.get("end")
    if t0 is None or t1 is None or t1 <= t0:
        return [(w, None) for w in words]
    if len(words) == 1:
        return [(words[0], float(t0))]
    span = float(t1 - t0)
    step = span / (len(words)-1)
    return [(w, float(t0) + i*step) for i, w in enumerate(words)]

def align_chords_to_words(segments, lrc_lines):
    segs = sorted(segments, key=lambda s: (s["start"], s["end"]))

    def chord_at(t):
        if t is None:
            return None
        for s in segs:
            if s["start"] <= t < s["end"]:
                return s["chord"]
        return None

    aligned = []
    for line in lrc_lines:
        words_with_t = interpolate_words(line)
        lyrics_text = " ".join(w for w, _ in words_with_t)
        if not lyrics_text:
            aligned.append({"lyrics": "", "placed": [], "mono": ""})
            continue

        placed = []
        col = 0
        last = None
        for idx, (w, t) in enumerate(words_with_t):
            ch = chord_at(t)
            if ch and ch != "N.C." and ch != last:
                placed.append((col, ch))
            last = ch
            col += len(w) + (1 if idx < len(words_with_t)-1 else 0)

        chord_row = [" "] * len(lyrics_text)
        for pos, ch in placed:
            for i, c in enumerate(ch):
                j = pos + i
                if 0 <= j < len(chord_row):
                    chord_row[j] = c

        aligned.append({
            "lyrics": lyrics_text,
            "placed": placed,
            "mono": "".join(chord_row) + "\n" + lyrics_text
        })

    return aligned

# === LOCAL PATHS ===
SONIC = r"/usr/local/bin/sonic-annotator/sonic-annotator-win64/sonic-annotator.exe"
VAMP_PATH = r"/usr/local/lib/vamp/"

VERSION = "SquareOne v1"
app = FastAPI(title=f"Chord Detector ({VERSION})")

def run_sonic_simplechord(wav_path: str) -> str:
    if not os.path.isfile(SONIC):
        raise RuntimeError(f"Sonic Annotator not found at: {SONIC}")
    env = os.environ.copy()
    env["VAMP_PATH"] = VAMP_PATH
    cmd = [
        SONIC,
        "-d", "vamp:nnls-chroma:chordino:simplechord",
        "-w", "lab", "--lab-stdout",
        wav_path.replace("\\", "/"),
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="ignore")
    if p.returncode != 0:
        raise RuntimeError("Sonic Annotator failed:\n" + (p.stderr or p.stdout or ""))
    return p.stdout

def merge_segments(segs, min_dur=0.6):
    if not segs:
        return segs
    merged = []
    cur = segs[0].copy()
    for s in segs[1:]:
        if s["chord"] == cur["chord"] and abs(s["start"] - cur["end"]) < 0.02:
            cur["end"] = s["end"]
        else:
            if (cur["end"] - cur["start"]) < min_dur and cur["chord"] != "N.C.":
                cur["chord"] = "N.C."
            merged.append(cur)
            cur = s.copy()
    if (cur["end"] - cur["start"]) < min_dur and cur["chord"] != "N.C.":
        cur["chord"] = "N.C."
    merged.append(cur)
    return merged

def is_wav_bytes(raw: bytes) -> bool:
    return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"

def write_wav_from_bytes(raw: bytes) -> str:
    if not is_wav_bytes(raw):
        raise RuntimeError("Please upload a WAV file (PCM/float). You can convert in any editor (e.g., Audacity).")
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    with open(path, "wb") as f:
        f.write(raw)
    return path

def synth_c_major_wav(seconds=3.0, sr=44100) -> str:
    t = np.linspace(0, seconds, int(sr*seconds), endpoint=False, dtype=np.float32)
    freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
    y = sum(0.33*np.sin(2*np.pi*f*t).astype(np.float32) for f in freqs)
    Nf = int(sr*0.02)
    fade = np.linspace(0, 1, Nf, dtype=np.float32)
    y[:Nf] *= fade; y[-Nf:] *= fade[::-1]
    y /= max(1e-9, np.abs(y).max())
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(path, y, sr)
    return path

# ----- Routes -----
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html>
<meta charset="utf-8"/>
<title>🎸 Tabslated - Guitar Tab Generator 🎸</title>
<body style="font-family:system-ui;background:#0b0f19;color:#e8eef8">
  <div style="max-width:900px;margin:2rem auto">
    <h1>🎸 Tabslated - Guitar Tab Generator 🎸</h1>
    <p style="opacity:.8">Upload a <strong>WAV</strong>. (Optional) paste <strong>LRC</strong> or plain lyrics. If you leave it empty and check “Auto-generate,” the server will transcribe vocals with Whisper.</p>

    <form id="f" enctype="multipart/form-data" style="display:grid;gap:0.75rem">
      <input type="file" name="file" accept=".wav" required />
      <label style="display:flex;gap:.5rem;align-items:center;opacity:.9">
        <input type="checkbox" id="auto" checked />
        Auto-generate lyrics with Whisper
      </label>
      <textarea name="lyrics_lrc" id="lrc" rows="8"
        placeholder="[00:12.50] First line of lyrics
[00:18.20] Next line of lyrics"
        style="width:100%;min-height:8rem;padding:.75rem;border-radius:8px;border:1px solid #23304b;background:#0e1422;color:#cfe4ff;font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"></textarea>
      <div style="display:flex;gap:.5rem">
        <button>Analyze + Align</button>
        <button type="button" id="smk">Run built-in smoketest</button>
      </div>
    </form>

    <h2 style="margin-top:2rem">Chord-over-Lyrics</h2>
    <pre id="tab" style="background:#0e1422;padding:1rem;border-radius:8px;white-space:pre-wrap;min-height:8rem">–</pre>
  </div>

<script>
const f = document.getElementById('f');
const tab = document.getElementById('tab');
const lrc = document.getElementById('lrc');
const auto = document.getElementById('auto');
const smk = document.getElementById('smk');

f.addEventListener('submit', async (e)=>{
  e.preventDefault();
  tab.textContent = 'Working…';

  const fd = new FormData(f);
  fd.append('lyrics_auto', auto.checked ? 'true' : 'false');

  try {
    const r = await fetch('/analyze', { method:'POST', body: fd });
    if (!r.ok) {
      tab.textContent = 'Error running /analyze';
      return;
    }
    const j = await r.json();

    if (j.aligned && j.aligned.length) {
      tab.textContent = j.aligned.map(x => x.mono).join("\\n\\n");
    } else if (j.lyrics_lrc) {
      tab.textContent = "No per-line alignment built, but LRC was returned:\\n\\n" + j.lyrics_lrc;
    } else {
      tab.textContent = "No lyrics provided/available. Paste LRC or enable auto-generate.";
    }
  } catch (err) {
    tab.textContent = 'Network/JS error calling /analyze';
  }
});

smk.addEventListener('click', async ()=>{
  tab.textContent='Running smoketest…';
  try {
    const r = await fetch('/_smoketest');
    if (!r.ok) {
      tab.textContent = 'Smoketest error.';
      return;
    }
    const j = await r.json();
    if (j.segments && j.segments.length) {
      tab.textContent = j.segments.map(s => `${s.start.toFixed(2)}–${s.end.toFixed(2)}\\t${s.chord}`).join("\\n");
    } else {
      tab.textContent = 'Smoketest returned no segments.';
    }
  } catch (e) {
    tab.textContent = 'Smoketest error.';
  }
});
</script>
</body>
"""
    return HTMLResponse(html)


@app.get("/__whoami")
def whoami():
    return {"version": VERSION, "sonic": SONIC, "vamp_path": VAMP_PATH, "cwd": str(pathlib.Path().resolve())}

@app.get("/health")
def health():
    env = os.environ.copy(); env["VAMP_PATH"] = VAMP_PATH
    try:
        out = subprocess.check_output([SONIC, "-l"], env=env, stderr=subprocess.STDOUT, text=True, errors="ignore")
        has_chordino = any("chordino" in ln.lower() for ln in out.splitlines())
        return {"ok": True, "has_chordino": bool(has_chordino), "note": "look for 'chordino' in plugins list"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/_smoketest")
def smoketest():
    wav = synth_c_major_wav()
    try:
        lab = run_sonic_simplechord(wav)
        segs = merge_segments(parse_lab(lab), min_dur=0.4)
        labels = sorted(set([s["chord"] for s in segs]))
        return {"ok": True, "labels": labels, "segments": segs, "raw_first_lines": lab.splitlines()[:8]}
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass

# --- helper: safe duration for spreading lyrics when no chords ---
def safe_audio_duration(path: str) -> float | None:
    try:
        info = sf.info(path)
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    return None


from fastapi import Form

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    lyrics_lrc: str | None = Form(default=None),
    lyrics_auto: str | None = Form(default=None),  # "true" to enable auto if lyrics missing
):
    wav = None
    try:
        # ---- Read upload (requires python-multipart installed) ----
        raw = await file.read()
        wav = write_wav_from_bytes(raw)  # raises if not a real WAV

        # ---- Try chord detection, but don't fail the whole request if it breaks ----
        segments_error = None
        final, pattern, sections = [], [], []
        try:
            lab = run_sonic_simplechord(wav)  # this might raise
            raw_segments = parse_lab(lab)
            final, pattern, sections = postprocess_variable_pattern(raw_segments)
        except Exception as e:
            segments_error = f"Chord detection unavailable: {type(e).__name__}: {e}"

        # ---- Key detection (optional; only if we have chords) ----
        det_key = None
        if final:
            try:
                tonic_pc, mode = estimate_key_krumhansl(wav)
                if tonic_pc is not None:
                    det_key = f"{PC_NAMES[tonic_pc]} {'major' if mode=='maj' else 'minor'}"
                    final = key_aware_correct(
                        final, tonic_pc, mode,
                        drop_out_of_scale_if_short=True, short_thresh=1.2
                    )
            except Exception:
                pass  # ignore key errors

        # ---- Lyrics: prefer user-provided; optionally auto with Whisper ----
        lrc_text = (lyrics_lrc or "").strip()
        asr_error = None
        if not lrc_text and (lyrics_auto or "").lower() == "true":
            try:
                # If your Whisper init is eager, consider lazy init to avoid long import stalls.
                lrc_text = whisper_to_lrc(wav, per_word=False)
            except Exception as e:
                asr_error = f"Auto-lyrics failed: {type(e).__name__}: {e}"
                lrc_text = ""  # continue without lyrics

        # ---- Align chords over lyrics (works even if final==[]) ----
        aligned = []
        if lrc_text:
            lines = parse_lrc(lrc_text)

            # If there are no timestamps at all, spread lines across duration
            if not any(ln.get("start") is not None for ln in lines):
                total = float(final[-1]["end"]) if final else 0.0
                # fallback to librosa duration estimate if no chord segments
                if total <= 0 and librosa is not None:
                    try:
                        y, sr = librosa.load(wav, sr=None, mono=True)
                        total = len(y) / float(sr)
                    except Exception:
                        total = 0.0
                n = max(1, len(lines))
                step = (total or n) / n
                for i, ln in enumerate(lines):
                    ln["start"] = i * step
                    ln["end"] = (i+1) * step if i < n-1 else total

            # fill missing ends from next start or song end
            song_end = float(final[-1]["end"]) if final else (lines[-1]["end"] if lines and lines[-1].get("end") else None)
            for i, ln in enumerate(lines):
                if ln.get("start") is not None and ln.get("end") is None:
                    ln["end"] = lines[i+1]["start"] if i+1 < len(lines) and lines[i+1].get("start") is not None else song_end

            aligned = align_chords_to_words(final, lines)

        return JSONResponse({
            "ok": True,
            "filename": file.filename,
            "detected_key": det_key,
            "detected_pattern": pattern,
            "sections": sections,
            "segments": final,            # may be []
            "segments_error": segments_error,  # string or None
            "lyrics_lrc": lrc_text or None,
            "aligned": aligned,           # list with .mono for display
            "asr_error": asr_error,       # string or None
        })

    except Exception as e:
        # Only true bad requests (e.g., not WAV) end up here
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            if wav and os.path.exists(wav):
                os.remove(wav)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    print("[startup] running:", pathlib.Path(__file__).resolve())
    uvicorn.run(app, host="127.0.0.1", port=8010, reload=False)
