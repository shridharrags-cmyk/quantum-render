"""motionlib.core — canvas, color, blending, post FX.

Pure numpy/PIL frame synthesis: no browser, no GPU, renders on CPU only.
All buffers are float32 RGBA in 0..1 until the final uint8 conversion.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu"

def hsl(h, s, l):
    """h in [0,1), s,l in [0,1] -> (r,g,b) float tuple."""
    h = h % 1.0
    if s == 0:
        return (l, l, l)
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    def conv(t):
        t = t % 1.0
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    return tuple(conv(h + f) for f in (1/3, 0, -1/3))

def hsl_rgb(h, s, l):
    """Vectorized HSL -> RGB. h,s,l arrays or scalars in 0..1 -> (...,3) or (3,)."""
    h = np.asarray(h, np.float32) % 1.0
    s = np.asarray(s, np.float32)
    l = np.asarray(l, np.float32)
    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    def conv(t):
        t = t % 1.0
        r = p + (q - p) * 6 * t
        r = np.where(t < 1/6, r, q)
        r = np.where(t < 1/2, r, p + (q - p) * (2/3 - t) * 6)
        r = np.where(t < 2/3, r, p)
        return r
    return np.stack([conv(h + 1/3), conv(h), conv(h - 1/3)], axis=-1)

def gradient(w, h, stops, angle=90):
    """Linear gradient. stops: [(pos0..1, (r,g,b)), ...]; angle degrees (90 = top->bottom)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle)
    d = (np.cos(a) * xx / max(w-1, 1) + np.sin(a) * yy / max(h-1, 1))
    d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
    pos = np.array([s[0] for s in stops], dtype=np.float32)
    cols = np.array([s[1] for s in stops], dtype=np.float32)
    out = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(d, pos, cols[:, c])
    return out

def screen(base, layer):
    return 1 - (1 - base) * (1 - layer)

def add(base, layer):
    return np.clip(base + layer, 0, 1)

def multiply(base, layer):
    return base * layer

def to_img(buf):
    return Image.fromarray((np.clip(buf, 0, 1) * 255).astype(np.uint8), "RGB")

def box_blur(buf, radius):
    """Fast separable box blur via cumsum (approximate gaussian)."""
    if radius < 1:
        return buf
    r = int(radius)
    out = buf
    for axis in (0, 1):
        c = np.cumsum(out, axis=axis, dtype=np.float32)
        n = out.shape[axis]
        pad_lo = np.take(c, [0] * r, axis=axis)
        pad_hi = np.take(c, [n - 1] * r, axis=axis)
        c = np.concatenate([pad_lo, c, pad_hi], axis=axis)
        win = 2 * r + 1
        out = (np.take(c, np.arange(r, n + r), axis=axis)
               - np.take(c, np.arange(0, n), axis=axis)) / win
    return out

def gaussian_blur(buf, radius):
    img = to_img(buf).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img).astype(np.float32) / 255.0

def grain(buf, amount, rng):
    return np.clip(buf + rng.normal(0, amount, buf.shape).astype(np.float32), 0, 1)

def vignette(buf, strength=0.35, power=2.2):
    h, w = buf.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx / w - 0.5) * 2) ** 2 + ((yy / h - 0.5) * 2) ** 2) / np.sqrt(2)
    mask = 1 - strength * np.clip(d, 0, 1) ** power
    return buf * mask[..., None]

def scanlines(buf, spacing=3, strength=0.06):
    h, w = buf.shape[:2]
    mask = 1 - strength * ((np.arange(h) % spacing) == 0).astype(np.float32)
    return buf * mask[:, None, None]

class Timeline:
    """Beat clock. bpm -> seconds."""
    def __init__(self, bpm, duration, fps):
        self.bpm, self.duration, self.fps = bpm, duration, fps
        self.beat = 60.0 / bpm
    def beat_f(self, t):
        return t / self.beat
    def bar_f(self, t, beats_per_bar=4):
        return t / (self.beat * beats_per_bar)
    def frames(self):
        return int(round(self.duration * self.fps))
    def quantize(self, t, div=4):
        """Snap time to nearest 1/div beat."""
        b = self.beat_f(t)
        return round(b * div) / div * self.beat