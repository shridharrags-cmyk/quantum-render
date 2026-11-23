"""
qfx.py — cinematic particle-field renderer (numpy + Pillow only)

The look: everything in the frame is made of dots (the reference's visual
language), but the dots live in real 3-D space with perspective, lighting,
depth of field, motion blur, bloom, anamorphic streaks and film grain.

Coordinate system: y up, z toward the viewer's back (camera looks down -z in
world terms; we build a view matrix and camera-space z is forward positive).
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

F32 = np.float32
TAU = math.pi * 2.0

# ----------------------------------------------------------------- utilities
def rad(d):
    return d * math.pi / 180.0

def clamp(x, a=0.0, b=1.0):
    return np.clip(x, a, b)

def smoothstep(a, b, x):
    t = clamp((np.asarray(x, dtype=F32) - a) / max(b - a, 1e-6))
    return t * t * (3.0 - 2.0 * t)

def sstep(a, b, x):
    return float(smoothstep(a, b, x))

def ease_out(x, p=3.0):
    x = clamp(x)
    return 1.0 - (1.0 - x) ** p

def ease_in(x, p=3.0):
    return clamp(x) ** p

def ease_io(x):
    return smoothstep(0, 1, x)

def lerp(a, b, t):
    return a + (b - a) * t

def mix(c0, c1, t):
    c0 = np.asarray(c0, F32); c1 = np.asarray(c1, F32)
    t = np.asarray(t, F32)
    if t.ndim: t = t[..., None]
    return c0 + (c1 - c0) * t

def norm(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-8)

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], F32)

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], F32)

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], F32)

def jitter(seed):
    return np.random.default_rng(seed)

# ---------------------------------------------------------------------- camera
def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, F32); target = np.asarray(target, F32)
    up = np.asarray(up, F32)
    f = norm(target - eye)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.array([1.0, 0.0, 0.0], F32)
    s = norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, f])          # rows: right, up, forward
    t = -R @ eye
    return R, t


class Cam:
    def __init__(self, W, H, eye, target, fov=42.0, roll=0.0, up=(0, 1, 0)):
        self.W, self.H = W, H
        R, t = look_at(eye, target, up)
        if abs(roll) > 1e-6:
            Rz = rot_z(roll)
            R = Rz @ R
            t = Rz @ t
        self.R, self.t = R.astype(F32), t.astype(F32)
        self.eye = np.asarray(eye, F32)
        self.f = (H * 0.5) / math.tan(rad(fov) * 0.5)
        self.fov = fov

    def project(self, P):
        P = np.asarray(P, F32)
        C = P @ self.R.T + self.t
        z = C[:, 2]
        inv = 1.0 / np.maximum(z, 1e-4)
        sx = self.W * 0.5 + self.f * C[:, 0] * inv
        sy = self.H * 0.5 - self.f * C[:, 1] * inv
        return sx, sy, z

    def depth(self, P):
        return np.asarray(P, F32) @ self.R[2] + self.t[2]


# ------------------------------------------------------------ point sculptors
def sphere_shell(n, r=1.0, thick=0.03, rng=None):
    rng = rng or np.random.default_rng(7)
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    p = np.stack([np.sin(phi) * np.cos(theta),
                  np.cos(phi),
                  np.sin(phi) * np.sin(theta)], 1)
    rr = r * (1.0 + rng.normal(0, thick, (n, 1)))
    return p * rr


def sphere_volume(n, r=1.0, rng=None):
    rng = rng or np.random.default_rng(11)
    v = rng.normal(size=(n, 3))
    v = norm(v) * (rng.random((n, 1)) ** (1.0 / 3.0)) * r
    return v


def torus(n, R=1.0, r=0.15, rng=None):
    rng = rng or np.random.default_rng(13)
    a = rng.random(n) * TAU
    b = rng.random(n) * TAU
    rr = r * (0.85 + 0.3 * rng.random(n))
    x = (R + rr * np.cos(b)) * np.cos(a)
    y = rr * np.sin(b)
    z = (R + rr * np.cos(b)) * np.sin(a)
    return np.stack([x, y, z], 1)


def wire_sphere(n_lat=9, n_lon=16, r=1.0, per=620):
    """Latitude circles + longitude meridians as dotted polylines."""
    out = []
    for i in range(1, n_lat):
        phi = math.pi * i / n_lat
        th = np.linspace(0, TAU, per, endpoint=False)
        out.append(np.stack([r * np.sin(phi) * np.cos(th),
                             np.full(per, r * np.cos(phi), F32),
                             r * np.sin(phi) * np.sin(th)], 1))
    for j in range(n_lon):
        th = TAU * j / n_lon
        phi = np.linspace(0, math.pi, per)
        out.append(np.stack([r * np.sin(phi) * np.cos(th), r * np.cos(phi),
                             r * np.sin(phi) * np.sin(th)], 1))
    return np.concatenate(out).astype(F32)


def disc(n, r=1.0, rng=None):
    rng = rng or np.random.default_rng(17)
    a = rng.random(n) * TAU
    rr = r * np.sqrt(rng.random(n))
    return np.stack([rr * np.cos(a), np.zeros(n), rr * np.sin(a)], 1)


def grid3(nx, nz, sx=1.0, sz=1.0, y=0.0):
    xs = (np.arange(nx) / max(nx - 1, 1) - 0.5) * sx
    zs = (np.arange(nz) / max(nz - 1, 1) - 0.5) * sz
    X, Z = np.meshgrid(xs, zs)
    return np.stack([X.ravel(), np.full(X.size, y, F32), Z.ravel()], 1)


def bez(p0, p1, p2, p3, t):
    t = np.asarray(t, F32)[:, None]
    u = 1 - t
    return (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3


def tube(points, n, radius=0.01, rng=None):
    """Scatter n points along a polyline, each in a disc perpendicular to it."""
    rng = rng or np.random.default_rng(19)
    P = np.asarray(points, F32)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    u = rng.random(n) * L[-1]
    i = np.clip(np.searchsorted(L, u) - 1, 0, len(P) - 2)
    f = ((u - L[i]) / np.maximum(seg[i], 1e-6))[:, None]
    base = P[i] * (1 - f) + P[i + 1] * f
    tang = norm(P[i + 1] - P[i])
    a = np.cross(tang, np.array([0.0, 1.0, 0.0], F32))
    a = norm(a + 0.001)
    b = np.cross(tang, a)
    ang = rng.random(n) * TAU
    rr = radius * np.sqrt(rng.random(n))
    off = a * (rr * np.cos(ang))[:, None] + b * (rr * np.sin(ang))[:, None]
    return base + off


# --------------------------------------------------------------- shading
def shade(P, N, base, lights, eye, amb=0.05, rim=0.7, spec=0.35, shin=40.0,
          fog=None, fog_density=0.06):
    """Blinn-Phong over a point cloud. lights = [(pos, color, intensity)]."""
    P = np.asarray(P, F32); N = norm(np.asarray(N, F32))
    base = np.asarray(base, F32)
    col = base[None, :] * amb
    V = norm(eye[None, :] - P)
    for pos, lc, li in lights:
        lc = np.asarray(lc, F32)
        Lv = np.asarray(pos, F32)[None, :] - P
        d = np.linalg.norm(Lv, axis=1, keepdims=True)
        L = Lv / np.maximum(d, 1e-6)
        att = li / (1.0 + 0.12 * d * d)
        ndl = clamp(np.sum(N * L, 1))[:, None]
        col = col + base[None, :] * lc[None, :] * ndl * att
        H = norm(L + V)
        ndh = clamp(np.sum(N * H, 1))[:, None] ** shin
        col = col + lc[None, :] * spec * ndh * att
    fres = (1.0 - clamp(np.sum(N * V, 1)))[:, None] ** 3
    col = col + base[None, :] * rim * fres
    if fog is not None:
        z = np.linalg.norm(P - eye[None, :], axis=1)[:, None]
        f = 1.0 - np.exp(-fog_density * np.maximum(z - 1.0, 0.0))
        col = col * (1 - f) + np.asarray(fog, F32)[None, :] * f
    return np.clip(col, 0, 4.0)


# --------------------------------------------------------------- frame layers
class Scene:
    """Accumulates additive point layers, then composites them with per-layer
    defocus. Unique blur radii share one buffer, so scenes pay only for the
    depth bands they actually use."""

    def __init__(self, W, H, cam):
        self.W, self.H = W, H
        self.cam = cam
        self.layers = {}   # radius -> HxWx3 float32
        self.n = 0

    def _buf(self, r):
        if r not in self.layers:
            self.layers[r] = np.zeros((self.H, self.W, 3), F32)
        return self.layers[r]

    def dots(self, P, col, energy=1.0, blur=0.8, gain=1.0, cull=True):
        P = np.asarray(P, F32)
        col = np.asarray(col, F32)
        if col.ndim == 1:
            col = np.broadcast_to(col, P.shape).copy()
        sx, sy, z = self.cam.project(P)
        ok = z > 0.08
        if cull:
            ok &= (sx > -60) & (sx < self.W + 60) & (sy > -60) & (sy < self.H + 60)
        if not np.any(ok):
            return
        W, H = self.W, self.H
        ix = np.clip(sx[ok].astype(np.int32), 0, W - 1)
        iy = np.clip(sy[ok].astype(np.int32), 0, H - 1)
        idx = iy.astype(np.int64) * W + ix
        e = np.asarray(energy, F32).reshape(-1)
        if e.size == 1:
            e = np.full(idx.shape, float(e[0]), F32)
        elif e.size == P.shape[0] if P.ndim == 2 else 1:
            e = e[ok]
        else:
            e = np.full(idx.shape, float(e.max()), F32)
        c = col[ok]
        buf = self._buf(float(blur))
        flat = buf.reshape(-1, 3)
        n = W * H
        for ch in range(3):
            w = e * c[:, ch] * gain
            if np.all(w == 0):
                continue
            flat[:, ch] += np.bincount(idx, weights=w, minlength=n).astype(F32)

    def line_dots(self, a, b, n, col, energy=1.0, blur=0.8, gain=1.0, r=0.0):
        t = np.linspace(0, 1, n)[:, None]
        P = np.asarray(a, F32) * (1 - t) + np.asarray(b, F32) * t
        self.dots(P, col, energy, blur, gain)

    def compose(self, bloom=True, exposure=1.0):
        out = np.zeros((self.H, self.W, 3), F32)
        for r, buf in sorted(self.layers.items()):
            out += blur_rgb(buf, r)
        return out


def blur_rgb(buf, r):
    if r < 0.35:
        return buf
    # Gamma-encoded 8-bit path: fast C gaussian that keeps highlight detail
    # instead of clipping the core of a bright object to a flat blob.
    m = float(buf.max())
    if m <= 1e-5:
        return buf
    enc = np.sqrt(np.clip(buf / m, 0.0, 1.0))
    u8 = (enc * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8, "RGB").filter(ImageFilter.GaussianBlur(r))
    dec = (np.asarray(im, dtype=F32) / 255.0) ** 2
    return dec * m


def resize_f(arr, w, h, filt=Image.BILINEAR):
    """Resize float HxWx3 (or HxW) array through 8-bit-free PIL 'F' path."""
    if arr.ndim == 2:
        im = Image.fromarray(arr.astype(F32), "F")
        return np.asarray(im.resize((w, h), filt), dtype=F32)
    chans = [np.asarray(Image.fromarray(arr[..., c].astype(F32), "F").resize((w, h), filt), F32)
             for c in range(arr.shape[2])]
    return np.stack(chans, -1)


# -------------------------------------------------------------------- post FX
_GRAIN = None
_VIG = {}

def grain_tile(seed=3, n=1024):
    global _GRAIN
    if _GRAIN is None:
        rng = np.random.default_rng(seed)
        g = rng.normal(0, 1, (n, n)).astype(F32)
        g = (g + np.roll(g, 1, 0) + np.roll(g, 1, 1)) / 3.0
        _GRAIN = g.astype(F32)
    return _GRAIN


def vignette(H, W, power=1.9, strength=0.55):
    key = (H, W, power)
    if key in _VIG:
        return _VIG[key]
    yy, xx = np.mgrid[0:H, 0:W].astype(F32)
    cx, cy = W * 0.5, H * 0.5
    r = np.sqrt(((xx - cx) / (W * 0.5)) ** 2 + ((yy - cy) / (H * 0.5)) ** 2)
    v = np.clip(1.0 - strength * np.maximum(r - 0.35, 0) ** power, 0.0, 1.0)
    _VIG[key] = v[..., None].astype(F32)
    return _VIG[key]


def hblur_box(arr, k):
    """Horizontal box blur via cumulative sum (float, any channels)."""
    if k < 1:
        return arr
    H, W = arr.shape[:2]
    pad = np.pad(arr, ((0, 0), (k, k), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=1, dtype=np.float32)
    cs = np.concatenate([np.zeros_like(cs[:, :1]), cs], axis=1)
    return (cs[:, 2 * k:2 * k + W] - cs[:, :W]) / float(2 * k)


def bloom(img, strength=0.72, thresh=0.72, streak=0.13, streak_color=(0.55, 0.85, 1.0)):
    H, W = img.shape[:2]
    hw, hh = max(W // 2, 2), max(H // 2, 2)
    small = resize_f(img, hw, hh)
    bright = np.maximum(small - thresh, 0.0)
    b1 = resize_f(blur_rgb(bright, 2.2), W, H)
    b2 = resize_f(blur_rgb(bright, 7.0), W, H)
    b3 = resize_f(blur_rgb(bright, 18.0), W, H)
    out = img + (b1 * 0.60 + b2 * 0.34 + b3 * 0.22) * strength
    if streak > 0:
        q = resize_f(bright, max(W // 4, 2), max(H // 4, 2))
        q = hblur_box(q, max(int(W * 0.05), 2))
        sc = np.asarray(streak_color, F32)[None, None, :]
        out = out + resize_f(q * sc, W, H) * streak
    return out


def aces(x):
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)


def grade(img, exposure=1.0, gamma=(1.02, 1.0, 0.98),
          lift=(0.008, 0.012, 0.020), gain=(1.02, 1.0, 0.97)):
    x = np.maximum(img * exposure, 0.0)
    x = aces(x)
    x = np.clip(x, 0, 1)
    x = x ** np.asarray(gamma, F32)[None, None, :]
    x = x * np.asarray(gain, F32)[None, None, :] + np.asarray(lift, F32)[None, None, :]
    # split tone: cool shadows, warm high end
    luma = (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722)[..., None]
    x = x + (1 - luma) ** 2 * np.array([0.004, 0.010, 0.022], F32)[None, None, :]
    x = x + luma ** 2 * np.array([0.020, 0.006, -0.008], F32)[None, None, :]
    return np.clip(x, 0, 1)


def chromatic(img, amount=0.0016):
    """Radial RGB split on highlights — cheap lens realism."""
    if amount <= 0:
        return img
    H, W = img.shape[:2]
    r = img[..., 0]; g = img[..., 1]; b = img[..., 2]
    def scale(ch, s):
        im = Image.fromarray(ch.astype(F32), "F")
        w, h = int(W * s), int(H * s)
        im = im.resize((w, h), Image.BICUBIC)
        canvas = Image.new("F", (W, H), 0.0)
        canvas.paste(im, ((W - w) // 2, (H - h) // 2))
        return np.asarray(canvas, dtype=F32)
    return np.stack([scale(r, 1 + amount), g, scale(b, 1 - amount)], -1)


def add_grain(img, t=0.0, amount=0.016):
    H, W = img.shape[:2]
    g = grain_tile()
    n = g.shape[0]
    ox = int((t * 137.0) % n); oy = int((t * 271.0) % n)
    reps_y = H // n + 2; reps_x = W // n + 2
    tile = np.tile(np.roll(np.roll(g, ox, 0), oy, 1), (reps_y, reps_x))[:H, :W]
    luma = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722)[..., None]
    return np.clip(img + tile[..., None] * amount * (0.35 + 0.65 * (1 - luma)), 0, 1)


def finish(img, out_w, out_h, exposure=1.0, grain=0.016, vig=0.5, ca=0.0016, t=0.0):
    """HDR scene (any res) -> graded, grained, vignetted uint8 frame."""
    img = bloom(img)
    img = grade(img, exposure=exposure)
    img = resize_f(img, out_w, out_h, Image.LANCZOS)
    img = chromatic(img, ca)
    img = img + np.array([0.008, 0.012, 0.020], F32)[None, None, :]
    img = img * vignette(out_h, out_w, strength=vig)
    img = add_grain(img, t, grain)
    img = img + np.random.default_rng(int(t * 1000) % 99991).uniform(-0.6, 0.6, img.shape).astype(F32) / 255.0
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------- typography
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONTS = {
    "grot": "SpaceGrotesk.ttf",
    "mono": "JetBrainsMono.ttf",
    "serif": "InstrumentSerif-Regular.ttf",
    "serif_i": "InstrumentSerif-Italic.ttf",
}
_fcache = {}

def font(name, size, weight=None):
    key = (name, int(size), weight)
    if key not in _fcache:
        f = ImageFont.truetype(f"{FONT_DIR}/{FONTS[name]}", int(size))
        if weight is not None:
            try:
                f.set_variation_by_axes([weight])
            except Exception:
                pass
        _fcache[key] = f
    return _fcache[key]


def text_mask(text, name, size, weight=None, tracking=0.0):
    """Render one line with a consistent baseline and a transparent margin
    (so the glow has room). Returns (mask, ink_w, ink_h, ox, oy)."""
    f = font(name, size, weight)
    tr = tracking * size
    widths = [f.getlength(ch) for ch in text]
    total = sum(widths) + tr * max(len(text) - 1, 0)
    margin = int(size * 0.26) + 2
    W = int(total + 2 * margin) + 2
    H = int(size * 1.85)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    x = margin
    base = H * 0.74
    for ch, w in zip(text, widths):
        d.text((x, base), ch, font=f, fill=255, anchor="ls")
        x += w + tr
    a = np.asarray(im, F32) / 255.0
    ys, xs = np.nonzero(a > 0.012)
    if len(xs) == 0:
        return a, W, H, 0, 0
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    gm = int(size * 0.20) + 2
    X0, X1 = max(x0 - gm, 0), min(x1 + gm, W)
    Y0, Y1 = max(y0 - gm, 0), min(y1 + gm, H)
    return a[Y0:Y1, X0:X1], x1 - x0, y1 - y0, x0 - X0, y0 - Y0


def put_mask(frame, sprite, x, y, color, alpha=1.0, glow=0.45, glow_r=None,
             mode="add", reveal=None):
    """Composite a text sprite onto a float HxWx3 frame. (x, y) is the ink
    top-left; the sprite's transparent margin is compensated for."""
    if alpha <= 0.002:
        return
    if isinstance(sprite, tuple):
        mask, iw, ih, ox, oy = sprite
        x = x - ox
        y = y - oy
    else:
        mask = sprite
    H, W = frame.shape[:2]
    mh, mw = mask.shape
    x = int(round(x)); y = int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + mw, W), min(y + mh, H)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x].copy()
    if reveal is not None:
        p, feather = reveal
        w = max(int(mw * clamp(p)), 0)
        ramp = np.zeros(mw, F32)
        if p >= 0.999:
            ramp[:] = 1.0
        elif w > 0:
            ramp[:w] = 1.0
            fx = max(int(feather), 1)
            edge = np.linspace(0, 1, fx, endpoint=False)
            ramp[max(w - fx, 0):w] = edge[:min(fx, w)]
        m = m * ramp[x0 - x:x1 - x][None, :]
    a = (m * alpha)[..., None]
    c = np.asarray(color, F32)[None, None, :]
    reg = frame[y0:y1, x0:x1]
    if mode == "add":
        reg += c * a
    else:
        reg *= (1.0 - a)
        reg += c * a
    if glow > 0:
        r = glow_r if glow_r else max(mw, mh) * 0.035
        g = np.asarray(Image.fromarray((m * 255).astype(np.uint8), "L")
                       .filter(ImageFilter.GaussianBlur(max(r, 0.5))), F32) / 255.0
        frame[y0:y1, x0:x1] += c * (g[..., None] * glow * alpha)


def label(frame, text, x, y, alpha=1.0, size=20, color=(0.42, 0.85, 1.0),
          rule=True, weight=500):
    sp = text_mask(text, "mono", size, weight, tracking=0.20)
    w, h = sp[1], sp[2]
    put_mask(frame, sp, x, y, color, alpha * 0.95, glow=0.18, glow_r=size * 0.25)
    if rule:
        H, W = frame.shape[:2]
        ly = int(y + h * 1.55)
        if 0 <= ly < H:
            frame[ly:ly + 1, int(x):int(x + min(w * 0.5, 130))] += \
                np.asarray(color, F32) * 0.35 * alpha


def rule(frame, x0, x1, y, color=(0.4, 0.85, 1.0), alpha=0.4, thick=1):
    H, W = frame.shape[:2]
    y = int(y); x0 = int(x0); x1 = int(x1)
    if 0 <= y < H:
        frame[max(y, 0):min(y + thick, H), max(x0, 0):min(x1, W)] += \
            np.asarray(color, F32) * alpha


def corner_marks(frame, alpha=0.35, inset=44, length=18, color=(0.5, 0.8, 1.0)):
    H, W = frame.shape[:2]
    c = np.asarray(color, F32) * alpha
    for (x, y, dx, dy) in [(inset, inset, 1, 1), (W - inset, inset, -1, 1),
                           (inset, H - inset, 1, -1), (W - inset, H - inset, -1, -1)]:
        frame[y:y + dy * 1, x:x + dx * length] += c
        frame[y:y + dy * length, x:x + dx * 1] += c


def text_points(text, name, size, weight=None, tracking=0.05, n=60000,
                rng=None, thresh=0.5):
    """Sample a text mask into 3-D points (z=0 plane, centred). Used by the
    finale where the particle cloud assembles into the title."""
    rng = rng or np.random.default_rng(23)
    m = text_mask(text, name, size, weight, tracking)[0]
    ys, xs = np.nonzero(m > thresh)
    if len(xs) == 0:
        return np.zeros((n, 3), F32)
    pick = rng.integers(0, len(xs), n)
    px = xs[pick].astype(F32) + rng.random(n)
    py = ys[pick].astype(F32) + rng.random(n)
    # add a little depth so the word has body
    pz = rng.normal(0, size * 0.006, n)
    P = np.stack([px - m.shape[1] * 0.5, m.shape[0] * 0.5 - py, pz], 1)
    return P * (1.0 / size)


# ------------------------------------------------------------------ motion blur
def shutter_samples(t, fps=24, n=2, shutter=0.55):
    if n <= 1:
        return [t]
    span = shutter / fps
    return [t + (k / (n - 1) - 0.5) * span for k in range(n)]


# --------------------------------------------------------------- extra doodads
def dust(n, rng, box=(6, 3, 6), t=0.0, speed=0.02):
    p = (rng.random((n, 3)) - 0.5) * np.asarray(box, F32)
    p[:, 1] += np.sin(t * 0.3 + p[:, 0]) * 0.05
    p[:, 0] += np.sin(t * 0.17 + p[:, 2] * 2) * speed * 8
    return p


def dot_ground(cam, S, t, n=9000, y=-1.15, extent=9.0, col=(0.16, 0.42, 0.6),
               energy=0.5, blur=1.1, depth_fade=True):
    """The reference's dotted floor: a receding lattice of faint points."""
    zs = -np.arange(n)
    rng = np.random.default_rng(5)
    xs = (np.arange(n) % 90) / 89.0 - 0.5
    zz = (np.arange(n) // 90) / 89.0
    P = np.zeros((n, 3), F32)
    P[:, 0] = (np.arange(n) % 90) / 89.0 * extent - extent / 2
    P[:, 2] = (np.arange(n) // 90) / 89.0 * extent - extent * 0.75
    P[:, 1] = y
    P[:, 2] += (t * 0.12) % (extent / 90.0)
    d = cam.depth(P)
    e = np.full(n, energy, F32)
    if depth_fade:
        e *= np.clip(1.2 - d / 7.0, 0.03, 1.0)
    S.dots(P, np.asarray(col, F32), e, blur=blur)
