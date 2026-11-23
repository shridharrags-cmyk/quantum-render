"""motionlib.audio — procedural score synthesis (procedural-music skill).

Numpy oscillator banks + envelopes + biquad filters, offline render to WAV.
No DAW, no assets, CPU-cheap (renders far faster than realtime).
"""
import numpy as np
import wave, struct, subprocess, os

SR = 44100
midi = lambda m: 440.0 * 2 ** ((m - 69) / 12)

SCALES = {
    "minor":   [0, 2, 3, 5, 7, 8, 10],
    "major":   [0, 2, 4, 5, 7, 9, 11],
    "dorian":  [0, 2, 3, 5, 7, 9, 10],
    "penta":   [0, 2, 4, 7, 9],
    "lydian":  [0, 2, 4, 6, 7, 9, 11],
}

def scale_notes(root, name, octaves=2):
    s = SCALES[name]
    return [root + 12 * o + i for o in range(octaves) for i in s]

# ---------------- oscillators ----------------
def osc(freq, dur, kind="sine", sr=SR, detune=0.0):
    t = np.arange(int(dur * sr)) / sr
    if kind == "sine":
        return np.sin(2 * np.pi * freq * t)
    if kind == "saw":
        return 2 * (freq * t % 1) - 1
    if kind == "square":
        return np.sign(np.sin(2 * np.pi * freq * t))
    if kind == "tri":
        return 2 * np.abs(2 * (freq * t % 1) - 1) - 1
    if kind == "noise":
        return np.random.default_rng(int(freq)).normal(0, 1, len(t))
    raise ValueError(kind)

def adsr(n, a, d, s, r, sustain_level=0.7):
    env = np.ones(n) * sustain_level
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na, nd = min(na, n), min(nd, max(0, n - na))
    nr = min(nr, max(0, n - na - nd))
    env[:na] = np.linspace(0, 1, na) if na else 0
    env[na:na + nd] = np.linspace(1, sustain_level, nd) if nd else sustain_level
    env[na + nd:na + nd + nr] = np.linspace(sustain_level, 0, nr) if nr else 0
    env[na + nd + nr:] = 0
    return env

def place(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf): return
    seg = sig[:max(0, min(len(sig), len(buf) - i))]
    buf[i:i + len(seg)] += seg * gain

def lowpass(x, cutoff, sr=SR):
    """One-pole LP."""
    a = np.exp(-2 * np.pi * cutoff / sr)
    y = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = a * acc + (1 - a) * x[i]
        y[i] = acc
    return y

def highpass(x, cutoff, sr=SR):
    return x - lowpass(x, cutoff, sr)

def delay(buf, at, time_s, feedback=0.35, gain=0.5):
    i = int(at * SR)
    d = int(time_s * SR)
    for k in range(i, len(buf) - d):
        buf[k + d] += buf[k] * feedback
    return buf

def tanh_soft(x, drive=1.6):
    return np.tanh(x * drive) / np.tanh(drive)

def kick(dur=0.28, f0=120.0, f1=42.0):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(phase)
    env = np.exp(-t * 14)
    click = np.random.default_rng(7).normal(0, 1, len(t)) * np.exp(-t * 220) * 0.25
    return (body + click) * env

def snare(dur=0.22):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, len(t))
    body = np.sin(2 * np.pi * 190 * t) * 0.4
    env = np.exp(-t * 22)
    hp = highpass(noise, 1200)
    return (hp * 0.8 + body) * env

def hat(dur=0.09, open_=False):
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(11)
    n = rng.normal(0, 1, len(t))
    env = np.exp(-t * (14 if open_ else 55))
    return highpass(n, 7000) * env * 0.6

# ---------------- groovy synths ----------------
def bandpass(x, lo, hi, sr=SR):
    """Gentle band enclosure: lowpass(hi) minus lowpass(lo)."""
    return lowpass(x, hi, sr) - lowpass(x, lo, sr)

def rhodes(freq, dur, sr=SR, bark=3.0, decay=1.1):
    """FM electric-piano-ish: sine carrier + sine modulator (the 'bark'),
    fast amp decay. Bell-like tine on top, warm body underneath."""
    t = np.arange(int(dur * sr)) / sr
    env = np.exp(-t / (decay * 0.45))
    mod = np.sin(2 * np.pi * freq * bark * t) * np.exp(-t / (decay * 0.12))
    car = np.sin(2 * np.pi * freq * t + 2.2 * mod)
    return car * env

def finger_bass(freq, dur, sr=SR, glide_from=None):
    """Finger-style electric bass: saw + sine sub through an envelope-driven
    lowpass, with optional portamento glide into the note."""
    t = np.arange(int(dur * sr)) / sr
    if glide_from:
        f = glide_from + (freq - glide_from) * np.clip(t / 0.045, 0, 1)
        ph = 2 * np.pi * np.cumsum(f) / sr
        saw = 2 * (ph / (2 * np.pi) % 1) - 1
    else:
        saw = osc(freq, dur, "saw", sr)
    sub = np.sin(2 * np.pi * freq * t) * 0.7
    x = lowpass(saw + sub, 500)
    env = np.concatenate([np.linspace(0, 1, max(1, int(0.008 * sr))), np.exp(-np.arange(max(0, int(dur * sr) - int(0.008 * sr))) / max(dur * 0.4, 1e-3))])
    return tanh_soft(x, 1.8) * env

def pluck(freq, dur, sr=SR, bright=2400.0):
    """Wah-ish muted pluck: detuned saws through a decaying bandpass."""
    t = np.arange(int(dur * sr)) / sr
    body = osc(freq, dur, "saw", sr) + osc(freq * 1.005, dur, "saw", sr)
    bp = bandpass(body, 600, bright)
    env = np.exp(-t / max(dur * 0.28, 1e-3))
    return bp * env * 3.0

def shaker(dur=0.06, sr=SR, gain=0.5):
    t = np.arange(int(dur * sr)) / sr
    rng = np.random.default_rng(17)
    n = highpass(rng.normal(0, 1, len(t)), 4200)
    return n * np.exp(-t * 60) * gain

def clap(dur=0.16, sr=SR):
    """Layered clap: 3 noise bursts + tail."""
    rng = np.random.default_rng(23)
    out = np.zeros(int(dur * sr))
    for k, dly in enumerate((0.0, 0.011, 0.021)):
        n = int(0.03 * sr)
        i = int(dly * sr) + k * int(0.002 * sr)
        burst = rng.normal(0, 1, n) * np.exp(-np.arange(n) / sr * 90)
        out[i:i + n] += burst * (1.0 if k == 2 else 0.7)
    tail = int((dur - 0.03) * sr)
    i0 = int(0.03 * sr)
    out[i0:i0 + tail] += rng.normal(0, 1, tail) * np.exp(-np.arange(tail) / sr * 16) * 0.6
    return highpass(out, 900) * 0.8

def write_wav(buf, path, sr=SR, stereo=False, natural=False):
    # natural=True: peak-normalize (no tanh, no hard clip) — transients preserved
    b = buf.astype(np.float64) if natural else tanh_soft(buf) * 0.89
    if natural:
        b = b / max(1.0, float(np.abs(b).max()) / 0.98)
    if stereo and b.ndim == 2:
        ch = (np.clip(b, -1, 1) * 32767).astype(np.int16).T   # (2, N)
        inter = np.empty((ch.shape[1] * 2,), dtype=np.int16)
        inter[0::2] = ch[0, :]
        inter[1::2] = ch[1, :]
        pcm = inter
    else:
        pcm = (np.clip(b, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2 if (stereo and b.ndim == 2) else 1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())

def loudnorm(path, out, i=-14, tp=-2.0, lra=11):
    """Two-pass loudnorm: measure with ebur128, then apply linearly."""
    import re
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-af",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    err = p.stderr
    def last(pat):
        m = re.findall(pat, err)
        return float(m[-1]) if m else None
    meas_i   = last(r"\bI:\s+(-?[\d.]+)\s+LUFS")
    meas_tp  = last(r"\bPeak:\s+(-?[\d.]+)\s+dBFS")
    meas_lra = last(r"\bLRA:\s+(-?[\d.]+)\s+LU")
    meas_thr = last(r"\bThreshold:\s+(-?[\d.]+)\s+LUFS")
    if None not in (meas_i, meas_tp, meas_lra, meas_thr):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-af", (f"loudnorm=I={i}:TP={tp}:LRA={lra}:measured_I={meas_i}:measured_TP={meas_tp}"
                                f":measured_LRA={meas_lra}:measured_thresh={meas_thr}:linear=true,aresample=44100"),
                        "-ar", "44100", out], check=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-af", f"loudnorm=I={i}:TP={tp}:LRA={lra},aresample=44100",
                        "-ar", "44100", out], check=True)

# ================= NATURAL-SOUND SYNTHESIS =================
def piano_body(freq, dur, vel=0.8, sr=SR, stretched=0.0004):
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    npart = 10
    for k in range(1, npart + 1):
        fk = freq * k * (1 + stretched * k * k)
        decay = dur * (0.55 / (1 + 0.35 * (k - 1)))
        amp = (0.9 ** (k - 1)) * (0.35 + 0.65 * vel) ** (1 if k > 3 else 0.2)
        if k == 1: amp *= 1.0
        out += amp * np.sin(2 * np.pi * fk * t) * np.exp(-t / max(decay, 0.05))
    out += 0.35 * np.sin(2 * np.pi * freq * 1.0035 * t) * np.exp(-t / (dur * 0.6))
    out += 0.28 * np.sin(2 * np.pi * freq * 0.997 * t) * np.exp(-t / (dur * 0.55))
    rng = np.random.default_rng(int(freq * 7) % 99991)
    out += 0.11 * vel * rng.normal(0, 1, n) * np.exp(-t * 110)
    return out * (0.5 + 0.5 * vel) / npart * 3.2

def upright_bass(freq, dur, vel=0.8, sr=SR, slide_from=None):
    n = int(dur * sr)
    t = np.arange(n) / sr
    if slide_from:
        f = slide_from + (freq - slide_from) * np.clip(t / 0.09, 0, 1)
        ph = 2 * np.pi * np.cumsum(f) / sr
        body = np.sin(ph)
    else:
        body = np.sin(2 * np.pi * freq * t)
    body += 0.4 * np.sin(2 * np.pi * freq * 2.01 * t)
    band = bandpass(body, 60, 160)
    x = lowpass(body + band * 1.5, 620)
    rng = np.random.default_rng(int(freq * 3) % 99991)
    x += 0.10 * vel * rng.normal(0, 1, n) * np.exp(-t * 120)
    env = np.exp(-t / (dur * 0.45))
    return x * env * vel * 1.4

def brush_swirl(dur=1.6, sr=SR, gain=0.5):
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(5)
    grain = rng.normal(0, 1, n)
    grain = highpass(grain, 1800)
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 2.6 * t + 0.5)
    env = np.clip(np.sin(np.pi * t / dur) * 2.2, 0, 1)
    return grain * am * env * gain * 0.5

def soft_kick(dur=0.3, sr=SR, vel=0.7):
    t = np.arange(int(dur * sr)) / sr
    f = 95 * (55 / 95) ** (t / dur)
    body = np.sin(2 * np.pi * np.cumsum(f) / sr)
    rng = np.random.default_rng(9)
    click = rng.normal(0, 1, len(t)) * np.exp(-t * 200) * 0.04 * vel
    return (body * np.exp(-t * 9) + click) * vel

def rim_snare(dur=0.18, sr=SR, vel=0.5):
    t = np.arange(int(dur * sr)) / sr
    rng = np.random.default_rng(13)
    knock = np.sin(2 * np.pi * 210 * t) * np.exp(-t * 60)
    wire = highpass(rng.normal(0, 1, len(t)), 3500) * np.exp(-t * 90) * 0.35
    return (knock + wire) * vel

def room_ir(sr=SR, rt60=1.8, size=0.85, seed=21):
    n = int(rt60 * sr)
    rng = np.random.default_rng(seed)
    ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / sr * (6.9 / rt60))
    for dly, g in ((0.011, 0.5), (0.019, 0.42), (0.027, 0.35), (0.041, 0.28), (0.058, 0.2)):
        i = int(dly * sr)
        ir[i:i + 400] += g * rng.normal(0, 1, 400) * np.exp(-np.arange(400) / sr * 60)
    pred = int(0.006 * sr)
    ir[:pred] *= np.linspace(0, 1, pred)
    ir /= np.abs(ir).max()
    return ir.astype(np.float32)

def wet(sig, ir, mix=0.22, sr=SR):
    if len(sig) < 16:
        return sig
    n = len(sig) + len(ir) - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    S = np.fft.rfft(sig, nfft); H = np.fft.rfft(ir, nfft)
    y = np.fft.irfft(S * H, nfft)[:n]
    out = np.zeros(max(n, len(sig)))
    out[:len(sig)] = sig * (1 - mix)
    out[:n] += y * mix
    return out

def humanize(t, ms=14.0, rng=None):
    return t + (rng.normal(0, ms / 1000.0) if rng else np.random.default_rng().normal(0, ms / 1000.0))
