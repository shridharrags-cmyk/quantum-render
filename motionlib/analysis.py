"""motionlib.analysis — machine-listening metrics (the objective half of "ears").

What an engineer checks by ear, measured: tempo/grid, key, spectral balance,
per-band energy over time, loudness, arrangement dynamics, stereo width.
"""
import wave, subprocess
from itertools import chain
import numpy as np

SR = 44100

def load(path):
    w = wave.open(path)
    n = w.getnframes()
    ch = w.getnchannels()
    d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        d = d.reshape(-1, 2)
    else:
        d = np.stack([d, d], axis=1)
    w.close()
    return d  # (N,2)

def mono(d):
    return d.mean(axis=1)

# ---------------- rhythm ----------------
def tempo_and_grid(x, sr=SR, hop=512, win=1024):
    """Spectral-flux onsets -> IOI histogram tempo + grid alignment score."""
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    S = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1))
    flux = np.maximum(0, np.diff(S, axis=0)).sum(axis=1)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    onsets = []
    for i in range(1, len(flux) - 1):
        if flux[i] > 1.6 and flux[i] == max(flux[i-2:i+3]):
            onsets.append(i * hop / sr)
    if len(onsets) < 8:
        return {"bpm": None, "onsets": onsets}
    iois = np.diff(onsets)
    iois = iois[(iois > 0.2) & (iois < 1.2)]
    if len(iois) < 4:
        return {"bpm": None, "onsets": onsets}
    hist, edges = np.histogram(iois, bins=40, range=(0.2, 1.2))
    ioi = (edges[hist.argmax()] + edges[hist.argmax()+1]) / 2
    bpm = 60 / ioi
    while bpm < 70: bpm *= 2
    while bpm > 180: bpm /= 2
    step = 60 / bpm / 4
    err = np.abs([((t / step) - round(t / step)) * step for t in onsets])
    return {"bpm": round(bpm, 1), "onsets": onsets,
            "grid_alignment": round(float(np.mean(err < 0.045)), 3)}

# ---------------- harmony ----------------
def key_estimate(x, sr=SR, win=2048, hop=512):
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    A = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1))
    freqs = np.fft.rfftfreq(win, 1/sr)
    chroma = np.zeros(12)
    for m in range(12):
        f0 = 440 * 2 ** ((m - 9) / 12)
        chroma[m] = A[:, (freqs > f0/1.06) & (freqs < f0*1.06)].sum()
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    maj = np.array([1,0,1,0,1,1,0,1,0,1,0,1]); minr = np.array([1,0,1,1,0,1,0,1,1,0,1,0])
    best = max(chain(((float(np.roll(maj,r) @ chroma), "major", r) for r in range(12)),
                 ((float(np.roll(minr,r) @ chroma), "minor", r) for r in range(12))))
    return {"key": f"{names[best[2]]} {best[1]}", "confidence": round(best[0]/chroma.sum(), 2)}

# ---------------- timbre / balance ----------------
def band_balance(x, sr=SR):
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1/sr)
    tot = sp.sum() + 1e-9
    def e(lo, hi): return round(float(sp[(f>=lo)&(f<hi)].sum()/tot*100), 1)
    return {"sub_20-60": e(20,60), "bass_60-250": e(60,250), "lowmid_250-1k": e(250,1000),
            "mid_1k-4k": e(1000,4000), "high_4k-12k": e(4000,12000), "air_12k+": e(12000, 22050)}

def section_dynamics(x, sr=SR, sections=6):
    n = len(x)//sections
    out = []
    for i in range(sections):
        seg = x[i*n:(i+1)*n]
        rms = float(np.sqrt((seg**2).mean()))
        out.append(round(20*np.log10(max(rms, 1e-9)), 1))
    return out

def stereo_width(d):
    l, r = d[:,0], d[:,1]
    return round(float(np.abs(l-r).mean()/max(np.abs((l+r)/2).mean(), 1e-9)*100), 1)

def low_band_phase(d, sr=SR, cutoff=250):
    """Correlation of L vs R below cutoff — catches the real defect:
    time-offset low end (Haas gone wrong), not reverb decorrelation."""
    from motionlib.audio import lowpass
    l = lowpass(d[:,0], cutoff); r = lowpass(d[:,1], cutoff)
    corr = float(np.corrcoef(l, r)[0, 1])
    return round(max(corr, 0.0), 2)

def loudness(path):
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = p.stderr.strip().split("\n")[-14:]
    I = [l for l in tail if "I:" in l]
    P = [l for l in tail if "Peak:" in l]
    return {"integrated": I[-1].split("I:")[-1].split("LUFS")[0].strip() if I else "?",
            "true_peak": P[-1].split("Peak:")[-1].replace("dBFS","").strip() if P else "?"}

def report(path):
    d = load(path); x = mono(d)
    t = tempo_and_grid(x); k = key_estimate(x)
    return {
        "file": path,
        "duration_s": round(len(d)/SR, 2),
        "tempo": t, "key": k,
        "balance": band_balance(x),
        "section_rms_db": section_dynamics(x),
        "stereo_width_pct": stereo_width(d),
        "low_band_corr": low_band_phase(d),
        "loudness": loudness(path),
    }
def eq_fft(x, curve, sr=SR, block=1 << 18, overlap=1 << 13):
    """Linear-phase EQ via spectrum shaping, CHUNKED with overlap-add so the
    peak memory stays ~tens of MB on a 100s stereo file (Android OOM-safe)."""
    n = len(x)
    freqs = np.array([c[0] for c in curve], float)
    gains_db = np.array([c[1] for c in curve], float)
    logf, logg = np.log(np.maximum(freqs, 1)), gains_db
    hop = block - overlap
    out = np.zeros(n + block, np.float32)
    win = np.hanning(block).astype(np.float32)
    for start in range(0, n, hop):
        seg = np.zeros(block, np.float32)
        end = min(n, start + block)
        seg[:end - start] = x[start:end]
        X = np.fft.rfft(seg)
        f = np.fft.rfftfreq(block, 1 / sr)
        g = 10 ** (np.interp(np.log(np.maximum(f, 1.0)), logf, logg) / 20.0)
        y = np.fft.irfft(X * g, n=block).astype(np.float32)
        out[start:start + block] += y * win
    # normalize the overlap-add gain
    wsum = np.zeros(n + block, np.float32)
    for start in range(0, n, hop):
        wsum[start:start + block] += win
    return (out[:n] / np.maximum(wsum[:n], 1e-8)).astype(np.float32)
