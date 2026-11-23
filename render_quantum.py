"""QUANTUM - how quantum computing works. 192s explainer.

Visual DNA (extracted from two reference videos via video_dna.py):
dark near-black fields, flat vector graphics, blue/teal accents, kinetic text.
Render: numpy frames -> ffmpeg pipe -> h264 -> mux with narr_final.wav.
"""
import sys, json, subprocess, time
import numpy as np
sys.path.insert(0, ".")
from motionlib.core import (
    gradient, hsl, hsl_rgb,
    add, screen, box_blur,
    grain, vignette, scanlines,
)
from motionlib.motion import (
    seg, anim, cubic_out, expo_out, back_out, char_transform,
)
from motionlib.text import glyph_mask, text_line, place, glow
from motionlib.particles import (
    disk_mask, ring_mask, rings_batch, disks_batch, poly_mask,
    value_noise, curl,
)

W, H, FPS = 960, 540, 24
TL = json.load(open("/tmp/opencode/quantum/audio/timeline.json"))
BG = gradient(W, H, [(0.0, hsl(0.62, 0.55, 0.030)), (1.0, hsl(0.70, 0.60, 0.045))], angle=110)
INK = np.array(hsl(0.63, 0.18, 0.93))      # near-white ink
BLUE = np.array(hsl(0.60, 0.95, 0.60))      # primary accent
TEAL = np.array(hsl(0.50, 0.90, 0.55))      # secondary accent
GREEN = np.array(hsl(0.38, 0.85, 0.55))     # qubit accent
DIM = np.array(hsl(0.62, 0.30, 0.45))
rng = np.random.default_rng(7)

def T(sec):
    return sec * FPS

def text(buf, s, size, cx, cy, color=INK, font="bold", alpha=1.0, tracking=-0.02, spacing=-0.02):
    m = text_line(s, size, font)[0]
    mm = place(m, W, H, cx, cy, alpha=alpha)
    return add(buf, mm[..., None] * color)

def kinetic(buf, s, size, cx, cy, t, dur, color=INK, stagger=0.03, font="bold"):
    """Per-character entrance from below."""
    chars = list(s)
    n = len(chars)
    widths = [glyph_mask(c, size, font).shape[1] for c in chars]
    xs, x = [], 0
    for w_ in widths:
        xs.append(x); x += w_ + int(size * -0.02)
    total = max(1, x)
    for i, c in enumerate(chars):
        p = char_transform(t - i * stagger, 0, n)
        if p <= 0.001:
            continue
        e = expo_out(p)
        dy = (1 - back_out(p)) * 60
        m = place(glyph_mask(c, size, font), W, H, cx - total / 2 + xs[i] + widths[i] / 2, cy + dy,
                  alpha=e)
        buf = add(buf, m[..., None] * color)
    return buf

def fade(buf, t, dur, total):
    """Scene-level fade in/out."""
    a = min(1.0, t / min(0.6, dur * 0.2)) * min(1.0, (dur - t) / min(0.6, dur * 0.2))
    return buf * a

# ---------------- SCENES ----------------
def sc_hook(buf, lt, dur):
    """Black field; a single glowing qubit; title assembles from drifting dots."""
    cx, cy = W / 2, H * 0.42
    # drifting dust
    n = 240
    px = (np.sin(np.arange(n) * 12.9898 + lt * 0.4) * 0.5 + 0.5) * W
    py = (np.cos(np.arange(n) * 78.233 + lt * 0.3) * 0.5 + 0.5) * H
    specs = [(px[i], py[i], 1.6) for i in range(0, n, 6)]
    buf = add(buf, disks_batch((H, W), specs)[..., None] * INK * 0.05)
    # the qubit dot + pulse rings (beat ~0.9s)
    phase = (lt / 0.9) % 1.0
    pulse = max(0.0, 1.0 - phase * 1.8)
    d = disk_mask((H, W), cx, cy, 10 + 26 * pulse, soft=6)
    buf = add(buf, box_blur(d[..., None], 6) * GREEN * (0.5 + 0.5 * pulse))
    buf = add(buf, d[..., None] * GREEN * (0.9 + 0.5 * pulse))
    ring = ring_mask((H, W), cx, cy, 60 + 220 * phase, thickness=2.0, soft=1) * (0.35 * (1 - phase))
    buf = add(buf, ring[..., None] * TEAL)
    # title: letters fly in from scattered positions
    title = "QUANTUM"
    size = 96
    chars = list(title)
    widths = [glyph_mask(c, size).shape[1] for c in chars]
    xs, x = [], 0
    for w_ in widths:
        xs.append(x); x += w_ + 6
    total = x
    for i, c in enumerate(chars):
        p = anim(lt, 1.2 + i * 0.16, 0.9, back_out)
        if p <= 0.001:
            continue
        ox = (hash(i * 7 + 3) % 100 / 100 - 0.5) * 500 * (1 - p)
        oy = (hash(i * 7 + 11) % 100 / 100 - 0.5) * 200 * (1 - p)
        m = place(glyph_mask(c, size), W, H, cx - total / 2 + xs[i] + widths[i] / 2 + ox,
                  cy - 10 + oy, alpha=p)
        buf = add(buf, box_blur(m[..., None], 4) * BLUE * 0.55 * p)
        buf = add(buf, m[..., None] * INK)
    kinetic(buf, "how quantum computing works", 34, cx, cy + 150, lt - 3.2, 1.2, DIM, stagger=0.02)
    return buf

def sc_bits(buf, lt, dur):
    """A field of classical bits; zoom toward one; it shrinks to an atom; the grid jitters."""
    cols, rows = 16, 9
    cw, ch = 52, 48
    gx0, gy0 = (W - cols * cw) / 2, (H - rows * ch) / 2 + 6
    zoom = anim(lt, 9.0, 5.0, cubic_out)
    sc = 1.0 + zoom * 1.8
    for r in range(rows):
        for c in range(cols):
            cx = gx0 + c * cw + cw / 2
            cy = gy0 + r * ch + ch / 2
            if zoom > 0.02 and abs(cx - W / 2) < 260 and abs(cy - H / 2) < 260:
                continue  # zoomed region dissolves
            flick = (hash(c * 31 + r * 17) % 100) < 50
            col = np.array(hsl(0.60, 0.55, 0.55)) if not flick else np.array(hsl(0.60, 0.9, 0.65))
            m = poly_mask((H, W), cx, cy, cw * 0.42, 4, rot=np.pi / 4, width=2)
            bit = "1" if flick else "0"
            if zoom > 0.3:
                bit = ""
            buf = add(buf, m[..., None] * col * 0.55)
            if bit:
                b = place(glyph_mask(bit, 26), W, H, cx, cy, alpha=0.95)
                buf = add(buf, b[..., None] * INK * 0.85)
    # zoomed core: square -> atom lattice
    zc = W / 2
    core = min(1.0, zoom / 0.9)
    if core > 0:
        s = 150 * (1 - core)
        m = poly_mask((H, W), zc, H / 2, max(s, 3), 4, rot=np.pi / 4, width=2)
        buf = add(buf, m[..., None] * TEAL)
        # atoms
        na = 9
        for k in range(na):
            a = k * 2.399 + lt
            rr = 10 + 40 * core * (0.5 + 0.5 * np.sin(lt * 2 + k))
            buf = add(buf, disk_mask((H, W), zc + rr * np.cos(a), H / 2 + rr * np.sin(a),
                                     2.5 + 1.5 * core, soft=1) * GREEN)
    jitter = np.sin(lt * 22) * 0.6 if lt > 10 else 0.0
    return np.roll(buf, int(jitter), axis=1)

def sc_turn(buf, lt, dur):
    """The grid dissolves into particles and swirls to a point."""
    n = 700
    px = (np.sin(np.arange(n) * 12.9898 + 5) * 0.5 + 0.5) * W
    py = (np.cos(np.arange(n) * 78.233 + 9) * 0.5 + 0.5) * H
    swirl = min(1.0, lt / (dur * 0.75))
    ang = np.arctan2(py - H / 2, px - W / 2) + lt * 1.2
    rad = np.sqrt((px - W / 2) ** 2 + (py - H / 2) ** 2) * (1 - swirl)
    sx = W / 2 + rad * np.cos(ang)
    sy = H / 2 + rad * np.sin(ang)
    specs = [(sx[i], sy[i], 1.8) for i in range(0, n, 3)]
    buf = add(buf, disks_batch((H, W), specs)[..., None] * BLUE * 0.5)
    bright = disk_mask((H, W), W / 2, H / 2, 14 + 30 * swirl, soft=8)
    buf = add(buf, box_blur(bright[..., None], 5) * GREEN * 0.9)
    kinetic(buf, "instead of fighting the strangeness", 40, W / 2, H * 0.16, lt - 0.8, 1.2, INK)
    kinetic(buf, "use it", 44, W / 2, H * 0.16 + 60, lt - 1.8, 1.0, TEAL)
    return buf

def sc_qubit(buf, lt, dur):
    """Electron orbital rings; a spin arrow that blurs until measured."""
    cx, cy = W / 2, H / 2 + 10
    # orbital ellipses
    for k, (rx, ry, col) in enumerate([(150, 60, BLUE), (170, 78, TEAL), (120, 150, DIM)]):
        m = ring_mask((H, W), cx, cy, 1, thickness=1.6, soft=1) * 0  # ellipse via poly
        pts = [(cx + rx * np.cos(a + lt * 0.4 * (1 if k % 2 else -1)),
                cy + ry * np.sin(a + lt * 0.4 * (1 if k % 2 else -1))) for a in np.linspace(0, 2 * np.pi, 90)]
        em = np.zeros((H, W), np.float32)
        from PIL import Image, ImageDraw
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).line(pts, fill=110, width=2)
        em = np.asarray(img).astype(np.float32) / 255.0
        buf = add(buf, em[..., None] * col * 0.6)
    # nucleus
    buf = add(buf, disk_mask((H, W), cx, cy, 9, soft=4)[..., None] * INK * 0.9)
    # spin arrow: blurred rotating state, snaps on 'measure'
    measured = lt > dur - 5.2
    ang = lt * 2.2
    if measured:
        ang = 0.0 if (hash(int(lt * 7)) % 2) else np.pi
    L = 110
    ax, ay = cx + L * np.cos(ang), cy + L * np.sin(ang)
    for k in range(6 if not measured else 1):
        a2 = ang if measured else ang + (k - 3) * 0.12
        x2, y2 = cx + L * np.cos(a2), cy + L * np.sin(a2)
        line = np.zeros((H, W), np.float32)
        from PIL import Image, ImageDraw
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).line([(cx, cy), (x2, y2)], fill=int(220 / (1 + k)), width=4)
        lm = np.asarray(img).astype(np.float32) / 255.0
        buf = add(buf, lm[..., None] * (GREEN if not measured else TEAL) * (0.9 if measured else 0.7))
    # equation + labels
    kinetic(buf, "spin up", 30, W * 0.16, H * 0.16, lt - 1.5, 0.8, DIM)
    kinetic(buf, "spin down", 30, W * 0.84, H * 0.16, lt - 2.3, 0.8, DIM)
    eq = "α|0⟩  +  β|1⟩"
    m = text_line(eq, 46)[0]
    buf = add(buf, place(m, W, H, W / 2, H * 0.86, alpha=min(1, lt / 2))[..., None] * INK)
    if measured:
        kinetic(buf, "measure → commit", 34, W / 2, H * 0.86 - 70, lt - (dur - 5.0), 0.7, TEAL)
    return buf

def sc_coin(buf, lt, dur):
    """A spinning coin: squash + shade + blur; it lands and commits."""
    cx, cy = W / 2, H / 2
    landed = lt > dur - 3.2
    land_p = anim(lt, dur - 3.2, 0.5, back_out)
    spin = 1 - land_p
    sx = abs(np.cos(lt * 7)) * spin
    rw = int(90 * max(sx, 0.08))
    # blur ghosts
    for k in (-2, -1, 0, 1, 2):
        rx = int(90 * abs(np.cos(lt * 7 + k * 0.35)) * spin)
        if rx > 4:
            m = disk_mask((H, W), cx, cy, rx, soft=10)
            buf = add(buf, box_blur(m[..., None], 4) * np.array(hsl(0.12, 0.85, 0.55)) * (0.12 / (abs(k) + 1)))
    m = disk_mask((H, W), cx, cy, max(rw, 6), soft=8)
    shade = gradient(W, H, [(0.0, hsl(0.11, 0.9, 0.45)), (1.0, hsl(0.09, 0.8, 0.25))], angle=60)
    buf = add(buf, box_blur(m[..., None], 2) * np.array(hsl(0.11, 0.9, 0.5)) * 0.55)
    buf = add(buf, m[..., None] * np.array(hsl(0.11, 0.9, 0.55)) * 0.9)
    if land_p > 0.5:
        kinetic(buf, "heads", 40, W * 0.30, cy, land_p - 0.5, 0.5, INK)
        kinetic(buf, "tails", 40, W * 0.70, cy, land_p - 0.5, 0.5, DIM)
    kinetic(buf, "not heads. not tails.", 38, W / 2, H * 0.14, lt - 0.7, 0.9, INK)
    kinetic(buf, "a blur of possibility", 38, W / 2, H * 0.14 + 54, lt - 1.6, 0.9, TEAL)
    return buf

def sc_explosion(buf, lt, dur):
    """States doubling: a filling grid; the 2^300 number racing."""
    cols, rows = 10, 6
    cw, chh = 84, 72
    gx0, gy0 = (W - cols * cw) / 2, (H - rows * chh) / 2 + 20
    phase = lt / (dur * 0.45)
    lit = int(2 * (2 ** min(phase * 6, 10)))
    lit = min(lit, cols * rows)
    for i in range(lit):
        r, c = i // cols, i % cols
        cx = gx0 + c * cw + cw / 2
        cy = gy0 + r * chh + chh / 2
        col = TEAL if i % 3 else GREEN
        m = poly_mask((H, W), cx, cy, cw * 0.40, 4, rot=np.pi / 4, width=2)
        age = lt - (phase * 6 - i / (cols * rows) * 6 - 6) * dur * 0.45 / 6
        a = 1.0 if lt > 8 else min(1.0, max(0.2, 1.0 - (8 - lt) * 0.2))
        buf = add(buf, m[..., None] * col * 0.6 * a)
    kinetic(buf, "states, not bits", 40, W / 2, H * 0.12, lt - 0.8, 0.9, INK)
    # the big number
    n = int(min(2 ** 300, 10 ** 89) ** (min(phase, 1) * 0.999) * 1.0) if phase < 1 else None
    if lt > dur * 0.45:
        big = "2³⁰⁰"
        m = text_line(big, 110)[0]
        buf = add(buf, box_blur(place(m, W, H, W / 2, H * 0.45, alpha=1) [..., None], 6) * BLUE * 0.6)
        buf = add(buf, place(m, W, H, W / 2, H * 0.45, alpha=1)[..., None] * INK)
        sub = "more than atoms in the observable universe"
        kinetic(buf, sub, 34, W / 2, H * 0.45 + 100, lt - dur * 0.45, 0.9, DIM)
    return buf

def sc_entangle(buf, lt, dur):
    """Two linked qubits; measuring one settles both."""
    ax, bx = W * 0.22, W * 0.78
    cy = H * 0.45
    pulse = (lt / 1.3) % 1.0
    # link
    for k in range(2):
        y = cy + (k - 0.5) * 26
        ring = ring_mask((H, W), (ax + bx) / 2, y, abs(bx - ax) / 2, thickness=2.2, soft=1) * (0.5 + 0.4 * pulse)
        buf = add(buf, ring[..., None] * BLUE)
    # travelling measure pulse
    if pulse < 0.6:
        px = ax + (bx - ax) * (pulse / 0.6)
        buf = add(buf, disk_mask((H, W), px, cy, 6 + 10 * pulse, soft=4)[..., None] * TEAL)
    for x, side in ((ax, "A"), (bx, "B")):
        d = disk_mask((H, W), x, cy, 16 + 8 * pulse, soft=6)
        buf = add(buf, box_blur(d[..., None], 4) * GREEN * 0.7)
        buf = add(buf, d[..., None] * GREEN)
        m = text_line(side, 40)[0]
        buf = add(buf, place(m, W, H, x, cy - 90, alpha=0.9)[..., None] * INK)
    kinetic(buf, "spooky action at a distance", 44, W / 2, H * 0.12, lt - 1.2, 1.0, INK)
    kinetic(buf, "— Einstein", 28, W / 2, H * 0.12 + 56, lt - 2.2, 0.8, DIM)
    kinetic(buf, "one problem. spread across every qubit.", 32, W / 2, H * 0.86, lt - dur + 6, 1.0, TEAL)
    return buf

def sc_interfere(buf, lt, dur):
    """Wave fronts from two sources; cancelling and reinforcing."""
    cy = H * 0.45
    src = [(W * 0.28, cy), (W * 0.72, cy)]
    phase = lt * 2.6
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wave = np.zeros((H, W), np.float32)
    for sx, sy in src:
        d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        wave += np.sin(d * 0.16 - phase * 6) * np.exp(-d / 620)
    wave /= 2.0
    # constructive vs destructive zones
    bright = np.clip(wave, 0, 1) ** 1.4
    dark = np.clip(-wave, 0, 1) ** 1.4
    buf = add(buf, bright[..., None] * np.array(hsl(0.52, 0.95, 0.55)) * 0.85)
    buf = add(buf, dark[..., None] * np.array(hsl(0.70, 0.60, 0.18)) * 0.45)
    for sx, sy in src:
        buf = add(buf, disk_mask((H, W), sx, sy, 10, soft=5)[..., None] * INK)
    kinetic(buf, "wrong answers cancel", 40, W * 0.30, H * 0.12, lt - 0.8, 0.9, np.array(hsl(0.70, 0.6, 0.55)))
    kinetic(buf, "the right answer reinforces", 40, W * 0.70, H * 0.12, lt - 2.0, 0.9, TEAL)
    if lt > dur - 6:
        kinetic(buf, "the needle is the only thing left standing", 36, W / 2, H * 0.86, lt - (dur - 6), 1.0, INK)
    return buf

def sc_cold(buf, lt, dur):
    """A cryogenic chamber; noise drifts in; the state collapses."""
    cold = gradient(W, H, [(0.0, hsl(0.60, 0.70, 0.05)), (0.6, hsl(0.58, 0.65, 0.09)), (1.0, hsl(0.55, 0.60, 0.03))], angle=90)
    buf = add(buf, cold * 0.55)
    # chamber rods
    for k in range(5):
        x = W * (0.16 + k * 0.17)
        buf = add(buf, poly_mask((H, W), x, H / 2, 150, 4, rot=0, width=1)[..., None] * DIM * 0.35)
    # the qubit glow, flickering
    flick = 0.5 + 0.5 * np.sin(lt * 3.1) * np.exp(-lt / 8)
    cx, cy = W / 2, H * 0.45
    d = disk_mask((H, W), cx, cy, 18 + 12 * flick, soft=8)
    buf = add(buf, box_blur(d[..., None], 6) * GREEN * 0.7 * flick)
    buf = add(buf, d[..., None] * GREEN * flick)
    # noise snow creeping in
    snow = (np.random.default_rng(int(lt * FPS)).random((H // 6, W // 6)))
    from PIL import Image
    sm = np.asarray(Image.fromarray((snow * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    creep = min(1.0, lt / (dur * 0.7))
    buf = add(buf, sm[..., None] * INK * 0.05 * creep)
    # error-correction chips gray out
    if lt > dur - 6:
        n = 8
        for k in range(n):
            x = W * 0.2 + k * (W * 0.6 / n)
            on = k < 3
            col = np.array(hsl(0.62, 0.05, 0.35)) if not on else GREEN
            buf = add(buf, poly_mask((H, W), x, H * 0.72, 12, 4, rot=np.pi / 4, width=2)[..., None] * col)
        kinetic(buf, "most of the machine fights the noise", 32, W / 2, H * 0.86, lt - (dur - 6), 0.8, DIM)
    kinetic(buf, "colder than deep space", 44, W / 2, H * 0.13, lt - 0.7, 1.0, BLUE)
    return buf

def sc_matters(buf, lt, dur):
    """Three application cards flip in."""
    cards = [("molecules", "new medicines", TEAL),
             ("encryption", "broken, rebuilt", BLUE),
             ("systems", "optimized", GREEN)]
    for i, (t1, t2, col) in enumerate(cards):
        p = anim(lt, 1.0 + i * 2.6, 0.8, back_out)
        if p <= 0.001:
            continue
        cx = W * (0.25 + 0.25 * i)
        cy = H * 0.45
        w_, h_ = 150 * p, 100 * p
        # card
        card = np.zeros((H, W), np.float32)
        y0, y1 = int(cy - h_), int(cy + h_)
        x0, x1 = int(cx - w_), int(cx + w_)
        if y1 > y0 and x1 > x0:
            card[y0:y1, x0:x1] = 1.0
        buf = add(buf, box_blur(card[..., None], 2) * np.array(hsl(0.62, 0.4, 0.12)) * 0.9)
        edge = ring_mask((H, W), cx, cy, max(w_, h_) * 0.72, thickness=2, soft=1)
        buf = add(buf, edge[..., None] * col * 0.6 * p)
        glyph = {"molecules": 6, "encryption": 4, "systems": 8}[t1]
        buf = add(buf, poly_mask((H, W), cx, cy - 18, 26 * p, glyph, rot=lt, width=2)[..., None] * col * p)
        m1 = text_line(t1, 30)[0]
        buf = add(buf, place(m1, W, H, cx, cy + 28, alpha=p)[..., None] * INK)
        m2 = text_line(t2, 22, "sans")[0]
        buf = add(buf, place(m2, W, H, cx, cy + 58, alpha=p)[..., None] * DIM)
    kinetic(buf, "not a faster computer", 44, W / 2, H * 0.13, lt - 0.8, 1.0, INK)
    kinetic(buf, "a different kind of machine", 34, W / 2, H * 0.13 + 56, lt - 1.8, 1.0, BLUE)
    return buf

def sc_outro(buf, lt, dur):
    """The single qubit returns, resolves, and stays."""
    cx, cy = W / 2, H * 0.45
    calm = min(1.0, lt / 3.0)
    d = disk_mask((H, W), cx, cy, 10 + 6 * calm, soft=6)
    buf = add(buf, box_blur(d[..., None], 6) * GREEN * 0.6 * calm)
    buf = add(buf, d[..., None] * GREEN * calm)
    ring = ring_mask((H, W), cx, cy, 70 + 20 * calm, thickness=2.0, soft=1) * 0.5 * calm
    buf = add(buf, ring[..., None] * TEAL)
    kinetic(buf, "a new language", 48, W / 2, H * 0.72, lt - 1.0, 1.0, INK)
    kinetic(buf, "spoken natively", 48, W / 2, H * 0.72 + 66, lt - 1.8, 1.0, TEAL)
    small = text_line("QUANTUM · how it works", 24, "cond")[0]
    buf = add(buf, place(small, W, H, W / 2, H * 0.90, alpha=0.5 * calm)[..., None] * DIM)
    return buf

SCENES = [(0.0, 12.0, sc_hook), (12.4, 33.4, sc_bits), (34.0, 37.4, sc_turn),
          (38.4, 60.6, sc_qubit), (61.4, 70.6, sc_coin), (71.4, 93.4, sc_explosion),
          (94.3, 118.2, sc_entangle), (118.9, 142.9, sc_interfere), (144.0, 160.2, sc_cold),
          (161.0, 179.6, sc_matters), (180.2, 192.0, sc_outro)]

def frame_at(t):
    buf = BG.copy()
    for t0, t1, fn in SCENES:
        if t0 <= t < t1:
            buf = fn(buf, t - t0, t1 - t0)
            break
    buf = vignette(buf, 0.42)
    buf = grain(buf, 0.012, np.random.default_rng(int(t * FPS) + 1))
    return buf

def render(out_mp4, audio):
    total = int(192.0 * FPS)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio, "-shortest",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
           "-c:a", "aac", "-b:a", "192k", out_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    from motionlib.core import to_img
    t0 = time.time()
    for i in range(total):
        t = i / FPS
        p.stdin.write(to_img(frame_at(t)).tobytes())
        if i % 120 == 0:
            el = time.time() - t0
            print(f"[quantum] {i}/{total} {el:.0f}s ~{(el/max(i+1,1))*(total-i):.0f}s left", flush=True)
    p.stdin.close(); p.wait()
    print("RENDER DONE", p.returncode)

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/opencode/quantum/out", exist_ok=True)
    render("/tmp/opencode/quantum/out/quantum_explainer.mp4", "/tmp/opencode/quantum/audio/narr_final.wav")
