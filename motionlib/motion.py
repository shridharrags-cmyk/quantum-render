"""motionlib.motion — easing, springs, kinetic helpers.

Timing philosophy follows design-motion-principles: motion should feel
physical. Springs over linear tweens for anything organic; overshoot only
where it reads as "alive"; exits faster than entrances.
"""
import numpy as np

# ---------------- easing ----------------
def clamp01(x):
    return min(1.0, max(0.0, x))

def cubic_out(x):
    x = clamp01(x); return 1 - (1 - x) ** 3
def cubic_in(x):
    x = clamp01(x); return x ** 3
def cubic_in_out(x):
    x = clamp01(x); return 4 * x * x * x if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def expo_out(x):
    x = clamp01(x); return 1 if x == 1 else 1 - 2 ** (-10 * x)
def quart_out(x):
    x = clamp01(x); return 1 - (1 - x) ** 4
def back_out(x, s=1.70158):
    x = clamp01(x); return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2
def elastic_out(x, p=0.4):
    x = clamp01(x)
    if x in (0, 1): return x
    return 2 ** (-10 * x) * np.sin((x - p / 4) * (2 * np.pi) / p) + 1
def bounce_out(x):
    x = clamp01(x)
    n, d = 7.5625, 2.75
    if x < 1 / d: return n * x * x
    if x < 2 / d: return n * (x - 1.5 / d) * x + 0.75
    if x < 2.5 / d: return n * (x - 2.25 / d) * x + 0.9375
    return n * (x - 2.625 / d) * x + 0.984375

def seg(t, t0, t1):
    """Normalized 0..1 progress of t across [t0,t1], clamped."""
    if t1 <= t0: return 1.0 if t >= t1 else 0.0
    return clamp01((t - t0) / (t1 - t0))

def anim(t, t0, dur, ease=cubic_out):
    return ease(seg(t, t0, t0 + dur))

# ---------------- spring ----------------
class Spring:
    """Critically-damped-ish spring, integrated semi-implicitly. Use for
    scale/position of anything that should feel physical."""
    def __init__(self, stiffness=180.0, damping=14.0, x0=0.0):
        self.k, self.c, self.x, self.v = stiffness, damping, x0, 0.0
    def set_target(self, value):
        self.target = value
    def step(self, dt, target):
        # substep for stability at 60fps-ish dt
        n = max(1, int(dt / (1 / 240)))
        h = dt / n
        for _ in range(n):
            a = -self.k * (self.x - target) - self.c * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x
    def value(self, dt, target):
        if not hasattr(self, "target") or target != getattr(self, "_last_target", None):
            self._last_target = target
        return self.step(dt, target)

# ---------------- kinetic helpers ----------------
def char_transform(t, i, n, mode="rise", spread=0.035):
    """Per-character progress offset for staggered kinetic type."""
    delay = i * spread
    return seg(t, delay, delay + 0.35)

def motion_blur_layer(prev, cur, decay=0.72):
    """Frame-difference trails: accumulate a decaying ghost of previous frame."""
    return prev * decay + cur * (1 - decay)

def beat_pulse(beat_phase, attack=0.06, release=0.35):
    """1.0 on the beat, decaying release — audio-reactive pulse envelope."""
    if beat_phase < attack:
        return 1.0 - beat_phase / attack * 0.0 + 0.0 if False else 1.0
    return max(0.0, 1.0 - (beat_phase - attack) / release)

def spectrum_ring(levels, idx, n_rings):
    """Map spectrum band -> ring radius scale."""
    band = levels[idx % len(levels)]
    return 0.85 + 0.5 * band
# ---------------- shape morphing (flubber-inspired) ----------------
def resample_polyline(pts, n):
    """Resample a polyline to exactly n evenly-spaced points (for morphing)."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return np.repeat(pts, n, axis=0)
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0, total, n)
    out = np.zeros((n, 2))
    for c in range(2):
        out[:, c] = np.interp(targets, cum, pts[:, c])
    return out

def morph_polylines(a, b, t, n=120):
    """Point-count-matched linear interpolation between two shapes.
    Precondition for smooth shape-morph transitions in numpy frames."""
    A, B = resample_polyline(a, n), resample_polyline(b, n)
    return A * (1 - t) + B * t

def morph_mask(shape, a, b, t, width=3, n=120):
    """Stroked morph between two polygon point lists -> float mask."""
    from PIL import Image, ImageDraw
    h, w = shape
    pts = morph_polylines(a, b, t, n)
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).line([tuple(p) for p in pts], fill=255, width=width, joint="curve")
    return np.asarray(img).astype(np.float32) / 255.0
