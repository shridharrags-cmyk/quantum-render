"""
film.py — QUANTUM, a 192 s cinematic explainer (v3 art direction).

Every pixel is a point of light in a 3-D scene: the reference videos' dotted
visual language, rendered with perspective, lighting, defocus, motion blur,
bloom, anamorphic streaks and film grain.

v3: the frame always has depth — a dotted data-grid sky, drifting bokeh stars,
a receding floor — and every shot is built around one legible hero object with
a light on its rim. No more empty black rectangles.
"""
import math
import os
import numpy as np
from qfx import (Cam, Scene, F32, TAU, rad, clamp, smoothstep, sstep, ease_out,
                 ease_in, ease_io, lerp, mix, norm, rot_x, rot_y, rot_z,
                 sphere_shell, sphere_volume, torus, disc, grid3, bez, tube,
                 wire_sphere, shade, blur_rgb, resize_f, bloom, grade, aces,
                 chromatic, add_grain, finish, vignette, text_mask, put_mask,
                 font, label, rule, corner_marks, text_points, shutter_samples,
                 dust, dot_ground, jitter)

W, H = 1280, 720          # scene render resolution (upscaled to 1080p)
OUT_W, OUT_H = 1920, 1080
FPS = 24
DUR = 192.0
NSUB = int(os.environ.get("QFX_NSUB", "2"))   # motion-blur sub-frames

# narration section starts (see audio/timeline.json)
SEC = [0.0, 12.514467, 34.107936, 38.530249, 61.560725, 71.462947,
       94.414784, 118.968798, 144.080091, 161.134059, 180.333243, DUR]

# palette — cold machine, warm consequence
CYAN   = np.array([0.30, 0.82, 1.00], F32)
TEAL   = np.array([0.12, 0.68, 0.72], F32)
ICE    = np.array([0.72, 0.92, 1.00], F32)
WHITE  = np.array([0.94, 0.98, 1.00], F32)
INDIGO = np.array([0.34, 0.38, 0.95], F32)
VIOLET = np.array([0.62, 0.38, 1.00], F32)
AMBER  = np.array([1.00, 0.58, 0.18], F32)
MINT   = np.array([0.55, 1.00, 0.82], F32)
LIME   = np.array([0.72, 1.00, 0.42], F32)
RED    = np.array([1.00, 0.30, 0.22], F32)
VOID   = np.array([0.010, 0.016, 0.030], F32)


# ------------------------------------------------------------------ helpers
def locate(t):
    t = clamp(t, 0.0, DUR - 1e-4)
    for i in range(len(SEC) - 1):
        if t < SEC[i + 1] or i == len(SEC) - 2:
            return i, t - SEC[i], SEC[i], SEC[i + 1] - SEC[i]
    return len(SEC) - 2, 0.0, 0.0, 1.0


def kf(u, keys):
    """Piecewise smooth interpolation of scalars or tuples."""
    if u <= keys[0][0]:
        return keys[0][1]
    for i in range(len(keys) - 1):
        a, va = keys[i]; b, vb = keys[i + 1]
        if u <= b:
            p = ease_io((u - a) / max(b - a, 1e-6))
            if isinstance(va, tuple):
                return tuple(x + (y - x) * p for x, y in zip(va, vb))
            return va + (vb - va) * p
    return keys[-1][1]


def fade(u, t0, t1, t2, t3):
    return sstep(t0, t1, u) * (1.0 - sstep(t2, t3, u))


def floor_pts(x0, x1, z0, z1, nx, nz, y, rng=None, jit=0.0):
    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(z0, z1, nz)
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X.ravel(), np.full(X.size, y, F32), Z.ravel()], 1).astype(F32)
    if jit and rng is not None:
        P[:, 0] += rng.normal(0, jit, P.shape[0])
        P[:, 2] += rng.normal(0, jit, P.shape[0])
    return P


def depth_energy(cam, P, base=0.5, near=1.0, far=9.0, floor=0.06):
    d = cam.depth(P)
    return base * np.clip(1.0 - (d - near) / (far - near), floor, 1.0)


def screen_glow(S, cx, cy, r, color, inten=1.0, power=2.4):
    H, W = S.H, S.W
    yy, xx = np.mgrid[0:H, 0:W].astype(F32)
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
    g = np.exp(-d2 * power)[..., None]
    S._buf(0.0)[...] += g * np.asarray(color, F32)[None, None, :] * inten


def stage(S, t, gx=0.62, gy=0.44, gcol=None, gsize=0.40, gint=0.16,
          stars=520, dustn=150, floor=None):
    """Common background: navy gradient, dot-matrix sky, bokeh stars, glow."""
    H, W = S.H, S.W
    yy = np.linspace(0.0, 1.0, H, dtype=F32)[:, None, None]
    top = np.array([0.030, 0.052, 0.120], F32)
    bot = np.array([0.004, 0.008, 0.020], F32)
    base = top * (1.0 - yy) + bot * yy
    S.base = np.broadcast_to(base, (H, W, 3)).astype(F32).copy()

    # dot-matrix paper, like the reference's dotted charts
    step = 16
    gy0, gx0 = np.mgrid[0:H:step, 0:W:step]
    grid = np.zeros((H, W), F32)
    grid[gy0.ravel(), gx0.ravel()] = 1.0
    fadey = (1.0 - 0.55 * yy)
    S.base += grid[..., None] * np.array([0.020, 0.034, 0.070], F32) * fadey
    # brighter rule every 8th row, faint
    S.base[::step * 8, :, :] += np.array([0.012, 0.020, 0.042], F32)

    if gcol is not None:
        screen_glow(S, W * gx, H * gy, W * gsize, gcol, gint)

    if stars:
        rng = np.random.default_rng(77)
        P = dust(stars, rng, box=(26.0, 13.0, 26.0), t=t, speed=0.02)
        P = P + np.array([0.0, 0.4, 0.0], F32)
        cam = S.cam
        d = cam.depth(P)
        ok = d > 1.2
        e = 0.30 * np.clip(1.25 - d / 22.0, 0.10, 1.0)
        col = mix(ICE, CYAN, rng.random(P.shape[0]) * 0.7)
        S.dots(P[ok], col[ok], e[ok], blur=2.6)
    if dustn:
        dust_layer(S, t, dustn, 2.6, energy=0.16)
    if floor is not None:
        f = dict(floor)
        dot_ground(S.cam, S, t, **f)


def dust_layer(S, t, n=260, blur=2.4, box=(9, 4, 9), energy=0.30, col=None):
    rng = np.random.default_rng(99)
    P = dust(n, rng, box=box, t=t, speed=0.03)
    d = S.cam.depth(P)
    e = energy * np.clip(1.35 - d / 9.0, 0.05, 1.0)
    S.dots(P, col if col is not None else ICE * 0.8, e, blur=blur)



def fib_sphere(n, R=1.0, jitter=0.0, rng=None):
    """Evenly spaced points on a sphere (golden-angle spiral) — reads as a
    dot-matrix surface rather than a fur of random speckle."""
    i = np.arange(n, dtype=F32) + 0.5
    ph = np.arccos(np.clip(1.0 - 2.0 * i / n, -1.0, 1.0))
    th = (math.pi * (3.0 - math.sqrt(5.0))) * i
    P = np.stack([np.cos(th) * np.sin(ph), np.cos(ph), np.sin(th) * np.sin(ph)], 1) * R
    if jitter and rng is not None:
        P = P + rng.normal(0, jitter, P.shape)
    return P.astype(F32)


def ring3d(n, R=1.0, r=0.02, tilt=0.0, yaw=0.0, rng=None):
    P = torus(n, R, r, rng)
    Rm = rot_y(yaw) @ rot_x(tilt)
    return P @ Rm.T


def glow_ball(S, c, r, col, energy=0.9, blur=0.9, n=40000, halo=3.0, back=0.10):
    rng = np.random.default_rng(1234)
    c = np.asarray(c, F32)
    P = fib_sphere(n, r, 0.004, rng) + c
    N = norm(P - c)
    cc = shade(P, N, col * 0.45, [((c[0] + 1.6, c[1] + 2.4, c[2] + 1.2), ICE, 2.2)], S.cam.eye,
               amb=0.10, rim=1.3)
    S.dots(P, cc, energy * face_energy(N, S.cam.eye, c, back), blur=0.0)
    S.dots(sphere_volume(n // 5, r * 1.7, rng) + c, col * 0.55, energy * 0.10, blur=blur + halo)


def lattice(nx, ny, nz, sx, sy, sz, rng=None, per=None):
    """Nodes plus every grid line of a 3-D box, sampled densely enough that
    each line reads as a dotted wire rather than a handful of specks."""
    xs = np.linspace(-sx * 0.5, sx * 0.5, nx)
    ys = np.linspace(-sy * 0.5, sy * 0.5, ny)
    zs = np.linspace(-sz * 0.5, sz * 0.5, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(F32)
    step = 0.020
    nxs = max(int(sx / step), 8)
    nys = max(int(sy / step), 8)
    nzs = max(int(sz / step), 8)
    segs = []
    for y in ys:
        for z in zs:
            segs.append(np.stack([np.linspace(xs[0], xs[-1], nxs), np.full(nxs, y),
                                  np.full(nxs, z)], 1))
    for x in xs:
        for z in zs:
            segs.append(np.stack([np.full(nys, x), np.linspace(ys[0], ys[-1], nys),
                                  np.full(nys, z)], 1))
    for x in xs:
        for y in ys:
            segs.append(np.stack([np.full(nzs, x), np.full(nzs, y),
                                  np.linspace(zs[0], zs[-1], nzs)], 1))
    edges = np.concatenate(segs).astype(F32)
    return nodes, edges


def face_energy(N, eye, C, back=0.10):
    """Dim the far hemisphere so a point sphere reads as solid, not glass."""
    V = norm(np.asarray(eye, F32)[None, :] - np.asarray(C, F32)[None, :])
    front = np.sum(np.asarray(N, F32) * V, 1) > -0.05
    return np.where(front, 1.0, back).astype(F32)


def braid(a, b, n, turns=3.0, amp=0.20):
    s = np.linspace(0.0, 1.0, n)
    a = np.asarray(a, F32); b = np.asarray(b, F32)
    base = a[None, :] * (1.0 - s)[:, None] + b[None, :] * s[:, None]
    d = norm(b - a)
    e1 = norm(np.cross(d, np.array([0.0, 1.0, 0.0], F32)))
    e2 = np.cross(d, e1)
    ph = turns * TAU * s
    p1 = base + amp * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
    p2 = base + amp * (np.cos(ph + math.pi)[:, None] * e1 + np.sin(ph + math.pi)[:, None] * e2)
    return p1, p2, s


# ==================================================================== 01. HOOK
def sc_hook(u, t):
    cam = Cam(W, H,
              eye=(kf(u, [(0, 0.55), (12, 0.30)]), kf(u, [(0, 0.34), (12, 0.18)]),
                   kf(u, [(0, 3.95), (12, 3.25)])),
              target=(kf(u, [(0, -0.70), (12, -0.66)]), 0.0, 0.0), fov=38,
              roll=rad(1.4) * math.sin(u * 0.34))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.64, gy=0.44, gcol=TEAL, gsize=0.34, gint=0.20)
    rng = jitter(1)
    Rm = rot_y(u * 0.30) @ rot_x(rad(16) + 0.06 * math.sin(u * 0.5))
    R = 0.90

    # electron cloud: a bright rim with a hollow, lit from the upper left
    Pc = fib_sphere(82000, R, 0.006, rng) @ Rm.T
    Nc = norm(Pc)
    cc = shade(Pc, Nc, TEAL * 0.30,
               [((-1.4, 2.2, 2.4), CYAN, 2.1), ((2.2, -0.6, 1.2), VIOLET, 0.9)],
               cam.eye, amb=0.10, rim=1.25, spec=0.35)
    S.dots(Pc, cc, 0.20 * face_energy(Nc, cam.eye, np.zeros(3, F32)), blur=1.0)

    # the model of the atom: three orbits + a wire shell
    for j, (tilt, yaw, energy) in enumerate([(rad(74), u * 0.42, 0.55),
                                             (rad(28), u * 0.42 + 2.1, 0.50),
                                             (rad(-52), u * 0.42 + 4.2, 0.46)]):
        orb = ring3d(5200, R * 1.10, 0.006, tilt=tilt, yaw=yaw, rng=rng)
        S.dots(orb, mix(CYAN, VIOLET, 0.25 * j), energy * 0.7, blur=0.0)
    Wp = wire_sphere(9, 12, R, 520) @ Rm.T
    S.dots(Wp, CYAN * 0.9, 0.34, blur=0.0)

    # electrons with comet trails
    for j in range(3):
        a0 = u * (1.1 + 0.17 * j) + j * 2.1
        tr = np.linspace(a0 - 1.15, a0, 520)
        rr = R * 1.10
        tilt = [rad(74), rad(28), rad(-52)][j]
        yaw = u * 0.42 + [0.0, 2.1, 4.2][j]
        Rm2 = rot_y(yaw) @ rot_x(tilt)
        P = np.stack([rr * np.cos(tr), np.zeros_like(tr), rr * np.sin(tr)], 1) @ Rm2.T
        fade_e = np.linspace(0.05, 1.0, len(tr)) ** 2
        S.dots(P, mix(CYAN, WHITE, 0.4), 0.95 * fade_e, blur=1.1)
        core = sphere_shell(3200, 0.035, 0.0, rng) + P[-1]
        S.dots(core, WHITE, 2.4, blur=0.8)
        S.dots(sphere_volume(1200, 0.14, rng) + P[-1], CYAN, 0.25, blur=3.0)

    # nucleus: a hot cluster
    S.dots(sphere_shell(26000, 0.070, 0.03, rng), WHITE, 1.15, blur=0.0)
    S.dots(sphere_volume(16000, 0.28, rng), CYAN * 0.8, 0.22, blur=3.4)
    S.dots(sphere_volume(4000, 0.09, rng), ICE, 0.9, blur=1.2)
    dust_layer(S, t, 120, 2.6, energy=0.14)
    return S


def ov_hook(f, u, t, dur):
    a = fade(u, 0.6, 1.8, 11.2, 12.4)
    label(f, "01 / THE MACHINE", 120, 96, a * 0.85, 18)
    big = fade(u, 2.6, 3.6, 6.0, 6.6)
    put_mask(f, text_mask("not faster.", "grot", 132, 300, -0.02),
             120, 470, WHITE, big, glow=0.5, reveal=(clamp((u - 2.6) / 0.9), 90))
    b2 = fade(u, 6.9, 7.7, 11.4, 12.2)
    put_mask(f, text_mask("different.", "grot", 132, 500, -0.025),
             120, 470, WHITE, b2, glow=0.55, reveal=(clamp((u - 6.9) / 0.8), 90))
    sub = fade(u, 7.6, 8.8, 11.4, 12.2)
    put_mask(f, text_mask("a different kind of machine", "grot", 34, 300, 0.01),
             124, 626, CYAN * 0.9, sub * 0.85, glow=0.25)


# =================================================================== 02. BITS
def sc_bits(u, t):
    fwd = kf(u, [(0, 0.0), (13, 2.2), (20.5, 3.6)])
    cam = Cam(W, H,
              eye=(kf(u, [(0, 0.30), (13, -0.10), (20.5, 0.10)]),
                   kf(u, [(0, 1.85), (13, 1.20), (20.5, 0.55)]),
                   kf(u, [(0, 3.60), (13, 2.80), (20.5, 1.55)])),
              target=(0.0, kf(u, [(0, 0.00), (13, -0.10), (20.5, -0.35)]), -1.6 - fwd),
              fov=52, roll=rad(0.7) * math.sin(u * 0.35))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.64, gy=0.40, gcol=AMBER * 0.5 + TEAL * 0.5, gsize=0.40, gint=0.13,
          floor=None)
    rng = jitter(2)

    # ---- switch field: cells of three dots, blinking to their own clock
    nx, nz = 30, 60
    xs = np.linspace(-4.6, 4.6, nx)
    zs = np.linspace(-9.0, 0.4, nz)
    X, Z = np.meshgrid(xs, zs)
    X = X.ravel(); Z = Z.ravel()
    y = -0.98
    phase = (X * 1.7 + Z * 0.9)
    on = 0.5 + 0.5 * np.sin(t * 2.4 + phase * 1.3)
    on = on ** 2
    # cell centre
    P0 = np.stack([X, np.full(X.size, y, F32), Z], 1)
    e0 = 0.22 + 1.55 * on
    S.dots(P0, mix(TEAL, CYAN, on), e0, blur=1.0)
    # gate dots either side
    for s in (-1, 1):
        P1 = P0 + np.array([s * 0.075, 0.0, 0.0], F32)
        S.dots(P1, mix(TEAL * 0.5, AMBER, on), 0.28 + 1.25 * on, blur=1.05)

    # rails down the field, and a bright horizon
    for x in (-4.75, 4.75):
        rp = np.stack([np.full(320, x), np.full(320, y), np.linspace(-9.2, 0.5, 320)], 1).astype(F32)
        S.dots(rp, TEAL * 0.7, 0.30, blur=1.3)
    hz = np.linspace(-6.5, 6.5, 900)
    S.dots(np.stack([hz, np.full(900, y), np.full(900, -9.4, F32)], 1), AMBER,
           0.30, blur=3.2)

    # ---- the dive: one switch grows into an atom
    dive = sstep(15.6, 19.4, u)
    if dive > 0.01:
        c = np.array([0.15, -0.62, -2.6], F32)
        S.dots(sphere_shell(int(26000 * dive) + 10, 0.10 + 0.34 * dive, 0.03, rng) + c,
               WHITE, 0.9 + 1.4 * dive, blur=0.8)
        if dive > 0.35:
            q = (dive - 0.35) / 0.65
            Wp = wire_sphere(7, 10, 0.12 + 0.44 * q, 420) + c
            S.dots(Wp, CYAN, 0.6 * q, blur=0.9)
            for j in range(2):
                orb = ring3d(9000, 0.16 + 0.46 * q, 0.008, rad(70) * (1 if j else -1),
                             u * 0.8 + j * 2.0, rng) + c
                S.dots(orb, CYAN if j == 0 else AMBER, 0.55 * q, blur=1.0)
    dust_layer(S, t, 150, 3.0, energy=0.16)
    return S


def ov_bits(f, u, t, dur):
    a = fade(u, 0.5, 1.5, 20.0, 21.4)
    label(f, "02 / BITS", 120, 96, a * 0.85, 18)
    t1 = fade(u, 2.4, 3.4, 8.6, 9.6)
    put_mask(f, text_mask("on.  off.", "grot", 104, 300, -0.01),
             120, 500, WHITE, t1, glow=0.4, reveal=(clamp((u - 2.4) / 0.7), 80))
    put_mask(f, text_mask("tens of billions of switches", "grot", 30, 300),
             124, 622, CYAN * 0.85, t1 * 0.8, glow=0.2)
    t2 = fade(u, 10.2, 11.2, 16.4, 17.4)
    put_mask(f, text_mask("a wall.", "grot", 120, 500, -0.03),
             120, 470, WHITE, t2, glow=0.5, reveal=(clamp((u - 10.2) / 0.7), 90))
    put_mask(f, text_mask("the switches reach the size of atoms", "mono", 22, 400, 0.08),
             124, 620, AMBER * 0.9, t2 * 0.85, glow=0.2)
    t3 = fade(u, 18.6, 19.8, 21.0, 21.6)
    put_mask(f, text_mask("the rules break.", "serif_i", 64, None),
             118, 540, ICE, t3, glow=0.35)


# ================================================================= 03. USE IT
def sc_useit(u, t):
    cam = Cam(W, H, eye=(kf(u, [(0, 0.15), (3.6, -1.9)]), 0.30,
                         kf(u, [(0, 3.30), (3.6, 3.90)])),
              target=(0.0, 0.0, 0.0), fov=42, roll=rad(-2) * u / 3.6)
    S = Scene(W, H, cam)
    stage(S, t, gx=0.62, gy=0.46, gcol=CYAN, gsize=0.32, gint=0.22)
    rng = jitter(3)
    a = sstep(0.0, 0.75, u)
    q = ease_io(clamp((u - 0.55) / 2.0))

    # order condensing out of a single point
    nodes, edges = lattice(7, 7, 7, 1.75, 1.75, 1.75)
    R = np.linalg.norm(nodes, axis=1, keepdims=True)
    from_pt = norm(nodes) * (0.05 + 1.9 * (1 - q))
    P = from_pt * (1 - q) + nodes * q
    col = mix(VIOLET, CYAN, clamp(q))
    S.dots(P, col, 0.9 * (0.4 + 0.6 * q), blur=0.9)
    E = edges * (0.25 + 0.75 * q)
    S.dots(E, CYAN * 0.85, 0.55 * q + 0.10, blur=1.0)
    # hot core that seeds it
    S.dots(fib_sphere(2600, 0.045 + 0.03 * (1 - q), 0.003, rng), ICE, 0.48, blur=0.0)
    S.dots(sphere_volume(9000, 0.42, rng), CYAN * 0.7, 0.10, blur=3.0)
    # flying sparks on the front of the wave
    n = 1200
    Ps = sphere_shell(n, 0.25 + 2.5 * q, 0.05, rng)
    S.dots(Ps, ICE, 0.20 * (1 - q), blur=1.4)
    dust_layer(S, t, 120, 2.6, energy=0.16)
    return S


def ov_useit(f, u, t, dur):
    a = fade(u, 0.15, 0.5, 3.1, 3.6)
    put_mask(f, text_mask("use it.", "grot", 150, 700, -0.03),
             120, 430, WHITE, a, glow=0.7, reveal=(clamp((u - 0.15) / 0.5), 100))
    label(f, "03 / THE TURN", 120, 96, a * 0.85, 18)
    l2 = fade(u, 1.4, 2.2, 3.1, 3.6)
    put_mask(f, text_mask("stop fighting the strangeness", "mono", 21, 400, 0.09),
             124, 620, CYAN, l2 * 0.8, glow=0.2)


# ================================================================== 04. QUBIT
def spinning_coin(n, R, thick, rng):
    nf = n // 2
    a = rng.random(nf) * TAU; rr = R * np.sqrt(rng.random(nf))
    front = np.stack([rr * np.cos(a), rr * np.sin(a), np.full(nf, thick / 2)], 1)
    back = front.copy(); back[:, 2] *= -1
    nr = n - 2 * nf
    b = rng.random(nr) * TAU
    rim = np.stack([R * np.cos(b), R * np.sin(b), rng.uniform(-thick / 2, thick / 2, nr)], 1)
    P = np.concatenate([front, back, rim])
    N = np.zeros_like(P)
    N[:nf] = [0, 0, 1]; N[nf:2 * nf] = [0, 0, -1]
    N[2 * nf:] = norm(np.stack([rim[:, 0], rim[:, 1], np.zeros(nr)], 1))
    return P, N


def sc_qubit(u, t):
    # A: the Bloch sphere (0 - 8.7)
    # B: not up / not down / both (8.7 - 15.3)
    # C: the coin (15.3 - 23.0)
    if u < 15.3:
        cam = Cam(W, H,
                  eye=(kf(u, [(0, 0.30), (15, 0.55)]), kf(u, [(0, 0.24), (15, 0.34)]),
                       kf(u, [(0, 4.25), (15, 3.95)])),
                  target=(-0.62, 0.0, 0.0), fov=40, roll=rad(1.2) * math.sin(u * 0.3))
        S = Scene(W, H, cam)
        stage(S, t, gx=0.64, gy=0.44, gcol=TEAL, gsize=0.34, gint=0.18)
        rng = jitter(4)
        R = 1.00
        # the sphere as a wire model on a dark glass shell
        WS = wire_sphere(11, 16, R, 620)
        S.dots(WS, CYAN * 0.85, 0.30, blur=0.0)
        equator = ring3d(11000, R, 0.004, tilt=0, yaw=0, rng=rng)
        S.dots(equator, ICE * 0.8, 0.50, blur=0.0)

        # vertical axis
        S.dots(np.stack([np.zeros(200), np.linspace(-R * 1.35, R * 1.35, 200), np.zeros(200)], 1),
               ICE * 0.5, 0.35, blur=1.0)
        for yy in (-R, R):
            S.dots(sphere_shell(2200, 0.035, 0.0, rng) + np.array([0, yy, 0], F32), WHITE, 1.4, blur=0.9)

        # state vector: swings from up to the equator plane, then keeps tilting
        th = math.radians(lerp(12.0, 108.0, ease_io(clamp(u / 8.6))))
        ph = u * 0.55 + 0.4
        tip = np.array([R * math.sin(th) * math.cos(ph), R * math.cos(th),
                        R * math.sin(th) * math.sin(ph)], F32)
        line = np.linspace(0, 1, 900)[:, None] * tip[None, :]
        S.dots(line, ICE, 0.85, blur=0.85)
        S.dots(sphere_shell(3600, 0.045, 0.0, rng) + tip, WHITE, 2.0, blur=0.9)

        # probability fan: the whole band of states the vector could take
        if u > 8.0:
            q = sstep(8.0, 11.0, u)
            for k in range(9):
                th2 = math.radians(20 + k * 18)
                tipp = np.array([R * math.sin(th2) * math.cos(ph), R * math.cos(th2),
                                 R * math.sin(th2) * math.sin(ph)], F32)
                ln = np.linspace(0, 1, 300)[:, None] * tipp[None, :]
                S.dots(ln, mix(CYAN, AMBER, k / 8), 0.30 * q, blur=1.1)
        S.dots(sphere_volume(9000, 1.6, rng), TEAL, 0.035, blur=4.0)
        dust_layer(S, t, 120, 2.6, energy=0.15)
        return S

    # ---- C: the coin
    cam = Cam(W, H, eye=(0.25, 0.12, kf(u, [(15.3, 2.75), (23, 2.30)])),
              target=(-0.45, 0.0, 0.0), fov=40, roll=rad(1.0) * math.sin(u * 0.5))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.62, gy=0.45, gcol=AMBER * 0.35 + TEAL * 0.65, gsize=0.30, gint=0.20)
    rng = jitter(5)
    R, th = 0.86, 0.10
    spin = (u - 15.3) * 7.0
    P0, N0 = spinning_coin(120000, R, th, rng)
    for k, (da, en) in enumerate([(0.0, 1.0), (-0.22, 0.22), (-0.46, 0.10)]):
        Rm = rot_y(spin + da) @ rot_x(rad(16))
        P = P0 @ Rm.T
        N = N0 @ Rm.T
        col = shade(P, N, ICE * 0.42,
                    [((2.2, 2.6, 3.0), WHITE, 2.4), ((-2.4, 0.6, 1.6), AMBER, 1.5)],
                    cam.eye, amb=0.10, rim=0.9, spec=0.75, shin=60)
        S.dots(P, col, en * 0.95, blur=0.85 if k == 0 else 1.8)
    S.dots(sphere_volume(9000, 0.6, rng), AMBER, 0.10, blur=3.6)
    dust_layer(S, t, 120, 2.6, energy=0.16)
    return S


def ov_qubit(f, u, t, dur):
    a = fade(u, 0.4, 1.4, 7.4, 8.4)
    label(f, "04 / THE QUBIT", 120, 96, a * 0.85, 18)
    put_mask(f, text_mask("one electron.", "grot", 96, 300, -0.02),
             120, 480, WHITE, a, glow=0.45, reveal=(clamp((u - 0.4) / 0.8), 80))
    put_mask(f, text_mask("spin — always up or down", "mono", 21, 400, 0.08),
             124, 604, CYAN * 0.85, a * 0.8, glow=0.18)

    b = fade(u, 8.8, 9.7, 13.4, 14.4)
    put_mask(f, text_mask("not up.", "grot", 112, 300, -0.02),
             120, 452, WHITE, b, glow=0.5, reveal=(clamp((u - 8.8) / 0.6), 80))
    put_mask(f, text_mask("not down.", "grot", 112, 300, -0.02),
             120, 560, WHITE, fade(u, 10.2, 11.0, 13.4, 14.4), glow=0.5,
             reveal=(clamp((u - 10.2) / 0.6), 80))
    put_mask(f, text_mask("both.", "grot", 112, 700, -0.02),
             120, 668, AMBER, fade(u, 11.6, 12.3, 13.4, 14.4), glow=0.7,
             reveal=(clamp((u - 11.6) / 0.6), 80))

    d = fade(u, 19.7, 20.4, 22.4, 23.0)
    put_mask(f, text_mask("α|0⟩ + β|1⟩", "serif", 108, None, 0.01),
             118, 470, AMBER, d, glow=0.55, reveal=(clamp((u - 19.7) / 0.5), 70))
    put_mask(f, text_mask("until you look", "mono", 22, 400, 0.10),
             124, 620, ICE * 0.9, d * 0.8, glow=0.18)


# =============================================================== 05. EXPLODE
def sc_explode(u, t):
    cam = Cam(W, H, eye=(0.0, 0.10, kf(u, [(0, 4.95), (9.9, 5.70)])),
              target=(0.0, 0.0, 0.0), fov=44, roll=rad(2.5) * math.sin(u * 0.4))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.60, gy=0.46, gcol=INDIGO * 0.6 + CYAN * 0.4, gsize=0.44, gint=0.16)
    rng = jitter(5)

    m = int(min(3 + (u / 9.9) * 9, 11))
    nodes, edges = lattice(m, m, m, 2.1, 2.1, 2.1, per=12)
    S.dots(nodes, mix(CYAN, WHITE, 0.35), 1.45, blur=0.0)
    S.dots(edges, TEAL, 0.34, blur=0.0)
    # a faint interior volume so the box feels full, not empty
    n = int(1200 + 60000 * clamp((u - 0.4) / 6.0))
    Pi = (rng.random((n, 3)) - 0.5) * 1.94
    burst = sstep(6.9, 8.9, u)
    R = np.linalg.norm(Pi, axis=1, keepdims=True)
    Pi = Pi * (1.0 + burst * (1.9 * (0.30 + 0.70 * R)))
    Pi = Pi @ rot_y(u * 0.30).T
    d = np.linalg.norm(Pi, axis=1)
    col = mix(CYAN, AMBER, clamp(d / 2.4) * 0.45)
    hot = np.exp(-d * d * 0.55)
    S.dots(Pi, col, 0.030 + 0.20 * burst + 0.03 * math.sin(u * 5.0), blur=1.0)

    # radial filaments once exploded
    if burst > 0.01:
        for j in range(30):
            th = j * TAU / 30 + 0.2
            ph = math.sin(j * 1.7) * 0.9
            dv = np.array([math.cos(th) * math.cos(ph), math.sin(ph),
                           math.sin(th) * math.cos(ph)], F32)
            ln = np.linspace(0.3, 2.8, 520)
            fil = dv[None, :] * ln[:, None] + rng.normal(0, 0.02, (520, 3))
            S.dots(fil, mix(CYAN, AMBER, 0.3), 0.55 * burst, blur=1.1)
    S.dots(sphere_volume(4000, 0.05, rng), WHITE, 0.8, blur=1.6)
    dust_layer(S, t, 160, 2.9, energy=0.18)
    return S


def ov_explode(f, u, t, dur):
    a = fade(u, 0.2, 0.8, 8.8, 9.6)
    label(f, "05 / SUPERPOSITION", 120, 96, a * 0.85, 18)
    n = int(1 + clamp((u - 0.2) / 6.2) * 16)
    cnt = f"2^{n}" if n < 17 else "2^300"
    put_mask(f, text_mask(cnt, "mono", 76, 300, 0.02),
             120, 170, CYAN, a * 0.95, glow=0.4)
    put_mask(f, text_mask("states held at once", "mono", 20, 400, 0.12),
             124, 262, ICE * 0.7, a * 0.75, glow=0.15)
    b = fade(u, 5.6, 6.6, 8.8, 9.6)
    put_mask(f, text_mask("they don't grow.", "grot", 92, 300, -0.02),
             120, 520, WHITE, b, glow=0.45, reveal=(clamp((u - 5.6) / 0.7), 80))
    put_mask(f, text_mask("they explode.", "grot", 110, 600, -0.03),
             120, 620, WHITE, fade(u, 6.9, 7.6, 8.8, 9.6), glow=0.6,
             reveal=(clamp((u - 6.9) / 0.6), 80))


# ============================================================ 06. ENTANGLEMENT
def sc_entangle(u, t):
    ex = kf(u, [(0, 0.35), (11, 1.80), (17, -1.55), (23, 0.30)])
    ey = kf(u, [(0, 0.30), (11, 0.85), (23, 0.28)])
    ez = kf(u, [(0, 4.55), (11, 3.95), (23, 4.20)])
    cam = Cam(W, H, eye=(ex, ey, ez), target=(0.0, 0.02, 0.0), fov=40,
              roll=rad(2.0) * math.sin(u * 0.35))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.50, gy=0.46, gcol=TEAL, gsize=0.42, gint=0.14)
    rng = jitter(6)

    A = np.array([-1.42, 0.0, 0.0], F32)
    B = np.array([1.42, 0.0, 0.0], F32)
    pul = 0.5 + 0.5 * math.sin(u * 1.6)
    for c, base, rr in ((A, CYAN, 0.62), (B, INDIGO, 0.50)):
        glow_ball(S, c, rr, base, energy=0.55 + 0.18 * pul, blur=1.0, n=90000)

    # braided link
    p1, p2, s = braid(A, B, 30000, turns=3.2, amp=0.19)
    jitter_n = rng.normal(0, 0.008, p1.shape)
    p1 = p1 + jitter_n; p2 = p2 + jitter_n[::-1]
    pulse = (u * 0.55) % 1.0
    dist = np.minimum(np.abs(s - pulse), 1 - np.abs(s - pulse))
    e = 0.30 + 1.25 * np.exp(-(dist ** 2) / 0.0022)
    S.dots(p1, mix(CYAN, WHITE, 0.25), e, blur=0.95)
    S.dots(p2, mix(VIOLET, CYAN, 0.35), e[::-1] * 0.9, blur=0.95)

    # distance ruler, dotted
    y = -1.12
    rp = np.stack([np.linspace(-2.5, 2.5, 900), np.full(900, y), np.full(900, -0.5)], 1).astype(F32)
    S.dots(rp, TEAL * 0.8, 0.26, blur=1.2)
    for x in np.linspace(-2.5, 2.5, 11):
        tp = np.stack([np.full(70, x), np.full(70, y), np.linspace(-0.5, -0.38, 70)], 1).astype(F32)
        S.dots(tp, CYAN * 0.7, 0.34, blur=1.1)

    # measurement: both ends flash at once
    if u > 18.4:
        q = math.exp(-((u - 19.6) ** 2) / 0.09)
        for c in (A, B):
            S.dots(sphere_shell(24000, 0.70, 0.02, rng) + c, WHITE, 0.9 * q, blur=1.4)
            S.dots(sphere_volume(5000, 0.28, rng) + c, AMBER, 0.6 * q, blur=2.6)
        mid = np.linspace(-2.2, 2.2, 700)
        S.dots(np.stack([mid, np.full(700, 0.0), np.full(700, 0.0)], 1), ICE, 0.55 * q, blur=1.6)
    dust_layer(S, t, 140, 2.8, energy=0.16)
    return S


def ov_entangle(f, u, t, dur):
    a = fade(u, 0.5, 1.5, 7.0, 8.0)
    label(f, "06 / ENTANGLEMENT", 120, 96, a * 0.85, 18)
    put_mask(f, text_mask("two qubits.", "grot", 96, 300, -0.02),
             120, 470, WHITE, a, glow=0.45, reveal=(clamp((u - 0.5) / 0.8), 80))
    put_mask(f, text_mask("one shared state.", "grot", 96, 500, -0.02),
             120, 570, WHITE, fade(u, 3.2, 4.0, 7.0, 8.0), glow=0.5,
             reveal=(clamp((u - 3.2) / 0.8), 80))
    put_mask(f, text_mask("no matter the distance", "mono", 21, 400, 0.10),
             124, 660, CYAN * 0.9, fade(u, 4.4, 5.2, 7.0, 8.0) * 0.8, glow=0.15)

    b = fade(u, 19.9, 20.6, 22.6, 23.2)
    put_mask(f, text_mask("spooky action", "serif_i", 66, None),
             120, 470, ICE, b, glow=0.3)
    put_mask(f, text_mask("at a distance.", "serif_i", 66, None),
             120, 540, ICE, b, glow=0.3)
    put_mask(f, text_mask("measured — and decided — instantly", "mono", 20, 400, 0.06),
             124, 634, AMBER, fade(u, 20.4, 21.2, 22.6, 23.2) * 0.9, glow=0.2)


# ============================================================= 07. INTERFERENCE
def sc_interference(u, t):
    cam = Cam(W, H,
              eye=(kf(u, [(0, -0.15), (24, 0.35)]), kf(u, [(0, 1.95), (13, 2.20), (24, 1.90)]),
                   kf(u, [(0, 2.05), (24, 1.85)])),
              target=(kf(u, [(0, -0.10), (24, 0.05)]), -0.62, kf(u, [(0, -1.05), (24, -0.85)])),
              fov=50, roll=rad(0.6))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.55, gy=0.40, gcol=TEAL, gsize=0.46, gint=0.12)
    rng = jitter(7)

    n = 230
    P = floor_pts(-3.1, 3.1, -3.2, 1.5, n, n, 0.0)
    X, Z = P[:, 0], P[:, 2]
    s1 = np.array([-1.15, 0.0, -0.95], F32)
    s2 = np.array([1.15, 0.0, -0.95], F32)
    r1 = np.sqrt((X - s1[0]) ** 2 + (Z - s1[2]) ** 2) + 1e-4
    r2 = np.sqrt((X - s2[0]) ** 2 + (Z - s2[2]) ** 2) + 1e-4
    k = 7.4
    w = 1.35
    h = (np.sin(k * r1 - w * t) + np.sin(k * r2 - w * t)) * 0.5
    env = np.exp(-(((X) ** 2 + (Z + 0.8) ** 2)) * 0.10)
    hh = h * env
    P[:, 1] = -0.36 + 0.20 * hh
    crest = np.clip(np.abs(hh), 0, 1)
    energy = 0.055 + 0.85 * crest ** 1.5
    col = mix(TEAL * 0.55, LIME, crest ** 1.2)
    S.dots(P, col, energy, blur=0.95)

    # emitters
    for c, cc in ((s1, WHITE), (s2, AMBER)):
        S.dots(sphere_shell(3000, 0.030, 0.0, rng) + c, cc, 0.85, blur=0.0)
        S.dots(sphere_volume(2000, 0.16, rng) + c, cc, 0.08, blur=2.4)
    # expanding rings from each source
    for j in range(2):
        rad0 = (t * 0.55 + j * 0.9) % 2.6
        for c, cc in ((s1, CYAN), (s2, AMBER)):
            th = np.linspace(0, TAU, 700)
            ring = np.stack([c[0] + rad0 * np.cos(th), np.full(700, -0.35), c[2] + rad0 * np.sin(th)], 1)
            e = 0.35 * math.exp(-rad0 * 0.7)
            S.dots(ring, cc, e, blur=1.4)

    # second half: the constructive column stands up
    q = sstep(11.5, 15.0, u)
    if q > 0.01:
        colp = np.stack([np.full(1400, 0.0), np.linspace(-0.4, 1.5, 1400),
                         np.full(1400, -0.95)], 1).astype(F32)
        S.dots(colp, LIME, 0.75 * q, blur=1.1)
        S.dots(sphere_shell(30000, 0.09, 0.01, rng) + np.array([0.0, 1.55, -0.95], F32),
               WHITE, 1.8 * q, blur=1.0)
        # antinode markers along the centre
        for zz in np.linspace(-2.2, 1.2, 9):
            S.dots(np.array([[0.0, -0.30, zz]], F32), LIME, 0.9 * q, blur=1.2)
    dust_layer(S, t, 150, 2.9, energy=0.15)
    return S


def ov_interf(f, u, t, dur):
    a = fade(u, 0.5, 1.5, 10.0, 11.0)
    label(f, "07 / INTERFERENCE", 120, 96, a * 0.85, 18)
    put_mask(f, text_mask("wrong answers", "grot", 84, 300, -0.02),
             120, 470, WHITE, a, glow=0.4, reveal=(clamp((u - 0.5) / 0.8), 80))
    put_mask(f, text_mask("cancel out.", "grot", 84, 300, -0.02),
             120, 560, WHITE, fade(u, 2.6, 3.4, 10.0, 11.0), glow=0.4,
             reveal=(clamp((u - 2.6) / 0.8), 80))
    put_mask(f, text_mask("like waves silencing each other", "mono", 20, 400, 0.08),
             124, 646, CYAN * 0.85, a * 0.75, glow=0.15)

    b = fade(u, 12.0, 13.0, 21.4, 22.6)
    put_mask(f, text_mask("the right answer", "grot", 80, 300, -0.02),
             120, 452, WHITE, b, glow=0.45, reveal=(clamp((u - 12.0) / 0.8), 80))
    put_mask(f, text_mask("reinforces itself.", "grot", 80, 500, -0.02),
             120, 540, AMBER, fade(u, 13.6, 14.4, 21.4, 22.6), glow=0.6,
             reveal=(clamp((u - 13.6) / 0.8), 80))
    put_mask(f, text_mask("tune the universe until the needle stands alone", "mono", 19, 400, 0.05),
             124, 640, ICE * 0.8, fade(u, 15.0, 16.0, 21.4, 22.6) * 0.8, glow=0.15)


# ============================================================= 08. DECOHERENCE
def sc_decohere(u, t):
    cam = Cam(W, H, eye=(kf(u, [(0, 0.20), (25, 1.05)]), kf(u, [(0, 0.32), (25, 0.20)]),
                          kf(u, [(0, 3.85), (25, 4.05)])),
              target=(-0.55, 0.0, 0.0), fov=42, roll=rad(1.5) * math.sin(u * 0.28))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.64, gy=0.46, gcol=CYAN, gsize=0.36, gint=0.16)
    rng = jitter(8)

    collapse = sstep(7.2, 9.2, u)
    s = 1.0 - 0.90 * collapse
    hot = collapse ** 2
    base_col = mix(CYAN, RED, hot)

    # orbit rings, tucked inside the frame
    for j, tilt in enumerate((rad(72), rad(24), rad(-48))):
        R = 1.05 * (1.0 - 0.55 * collapse)
        orb = ring3d(7000, R, 0.006, tilt=tilt, yaw=u * (0.35 + 0.1 * j) + j * 1.7, rng=rng)
        S.dots(orb, mix(base_col, AMBER, 0.30 * j), 0.26 * (1 - 0.5 * collapse), blur=0.0)

    # the qubit itself
    R0 = 0.80 * s
    if R0 > 0.02:
        glow_ball(S, (0.0, 0.0, 0.0), R0, base_col, energy=0.65 + 0.3 * hot, blur=0.95,
                  n=max(int(90000 * max(s, 0.05)), 8000))
    # the point it collapses into
    S.dots(sphere_shell(9000, 0.035 + 0.02 * hot, 0.0, rng), WHITE, 2.2 * collapse + 0.4, blur=0.8)

    # shockwave
    if 0.02 < collapse < 1.0:
        rr = 1.0 + 1.9 * collapse
        th = np.linspace(0, TAU, 1200)
        ring = np.stack([rr * np.cos(th), np.zeros(1200), rr * np.sin(th)], 1).astype(F32)
        S.dots(ring, AMBER, 0.9 * (1 - collapse) ** 1.6, blur=2.0)

    # error-correction cage after the collapse
    q = sstep(9.0, 11.0, u)
    if q > 0.01:
        nodes, edges = lattice(5, 5, 5, 1.0, 1.0, 1.0, per=22)
        S.dots(edges, AMBER, 0.42 * q, blur=1.1)
        S.dots(nodes, ICE * 0.9, 0.7 * q, blur=0.9)
        for k in range(6):
            a = u * 0.6 + k * 1.05
            r = 0.62
            p = np.array([r * math.cos(a), 0.25 * math.sin(a * 1.7), r * math.sin(a)], F32)
            S.dots(sphere_shell(2500, 0.025, 0.0, rng) + p, CYAN, 0.9 * q, blur=1.0)
    dust_layer(S, t, 150, 2.8, energy=0.16)
    return S


def ov_decohere(f, u, t, dur):
    a = fade(u, 0.5, 1.5, 8.0, 9.0)
    label(f, "08 / DECOHERENCE", 120, 96, a * 0.85, 18)
    put_mask(f, text_mask("the moment you look", "grot", 72, 300, -0.02),
             120, 452, WHITE, a, glow=0.4, reveal=(clamp((u - 0.5) / 0.8), 80))
    put_mask(f, text_mask("it collapses.", "grot", 96, 600, -0.03),
             120, 520, WHITE, fade(u, 3.4, 4.2, 8.0, 9.0), glow=0.55,
             reveal=(clamp((u - 3.4) / 0.7), 90))
    put_mask(f, text_mask("superposition is fragile", "mono", 20, 400, 0.10),
             124, 646, CYAN * 0.85, fade(u, 4.6, 5.4, 8.0, 9.0) * 0.8, glow=0.15)

    b = fade(u, 12.2, 13.0, 19.4, 20.4)
    put_mask(f, text_mask("0.0476 mK", "mono", 60, 500, 0.02),
             120, 452, CYAN, b * 0.95, glow=0.4)
    put_mask(f, text_mask("colder than deep space", "mono", 20, 400, 0.10),
             124, 540, ICE * 0.75, b * 0.8, glow=0.15)
    c = fade(u, 15.2, 16.2, 19.4, 20.4)
    put_mask(f, text_mask("fragile", "serif_i", 92, None),
             120, 600, AMBER, c, glow=0.4)
    put_mask(f, text_mask("error correction eats most of the machine", "mono", 20, 400, 0.05),
             124, 690, ICE * 0.75, fade(u, 16.4, 17.2, 19.4, 20.4) * 0.8, glow=0.15)


# =================================================================== 09. APPLICATIONS
def sc_apps(u, t):
    seg = int(u // 5.67)
    v = u - seg * 5.67
    ez = (4.05, 4.35, 4.10)[min(seg, 2)]
    cam = Cam(W, H, eye=(kf(v, [(0, 0.55), (5.0, -0.75)]) * (1.0 if seg != 1 else 0.4),
                         0.26, ez),
              target=(0.0, 0.0, 0.0), fov=42, roll=rad(1.2) * math.sin(u * 0.5))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.62, gy=0.46, gcol=CYAN, gsize=0.34, gint=0.18,
          floor=dict(n=7000, y=-1.15, extent=8.0, col=TEAL * 0.5, energy=0.14, blur=1.8))
    rng = jitter(9 + seg)

    if seg == 0:
        # ---- a molecule: two atom clusters, three bonds
        A = np.array([-0.62, 0.0, 0.0], F32)
        B = np.array([0.62, 0.0, 0.0], F32)
        spin = u * 0.42
        Rm = rot_y(spin) @ rot_x(rad(12))
        PA = sphere_shell(62000, 0.48, 0.02, rng) + A
        PB = sphere_shell(46000, 0.38, 0.02, rng) + B
        PA = PA @ Rm.T; PB = PB @ Rm.T
        cA = shade(PA, norm(PA - A @ Rm.T), CYAN * 0.45,
                   [((1.2, 2.2, 2.6), ICE, 1.9), ((-2.0, 0.2, 1.0), VIOLET, 0.8)],
                   cam.eye, amb=0.10, rim=1.2)
        cB = shade(PB, norm(PB - B @ Rm.T), AMBER * 0.40,
                   [((1.2, 2.2, 2.6), ICE, 1.8), ((-2.0, 0.2, 1.0), AMBER, 0.9)],
                   cam.eye, amb=0.10, rim=1.2)
        S.dots(PA, cA, 0.62, blur=0.95); S.dots(PB, cB, 0.62, blur=0.95)
        for k in range(3):
            ph = k * TAU / 3 + 0.4
            off = np.array([0, 0.19 * math.sin(ph), 0.19 * math.cos(ph)], F32)
            bond = tube(np.stack([A @ Rm.T + off, B @ Rm.T + off]), 16000, 0.016, rng)
            S.dots(bond, TEAL, 0.55, blur=1.05)
    elif seg == 1:
        # ---- cryptography: an amber key against a lattice wall
        spin = u * 0.30
        Rm = rot_y(rad(-26) + 0.25 * math.sin(spin)) @ rot_x(rad(6))
        # the wall: a square grid of dotted lines, edge to edge
        g = 4.1
        xs = np.linspace(-g * 0.5, g * 0.5, 15)
        nline = 128
        wall = []
        for y in xs:
            wall.append(np.stack([np.linspace(xs[0], xs[-1], nline), np.full(nline, y),
                                  np.zeros(nline)], 1))
        for x in xs:
            wall.append(np.stack([np.full(nline, x), np.linspace(xs[0], xs[-1], nline),
                                  np.zeros(nline)], 1))
        wall = np.concatenate(wall).astype(F32)
        wall = wall - np.array([0.0, 0.0, 0.55], F32)
        wall = wall @ Rm.T
        nw = F32(min(1.0, 0.30 + 0.55 * sstep(0.0, 1.2, v)))
        S.dots(wall, mix(TEAL * 0.85, CYAN, 0.45), 0.42 * nw, blur=0.0)

        # the key: bow ring, shaft, two teeth — one continuous warm object
        K = np.array([0.62, 0.12, 0.55], F32)
        kR = rot_y(rad(8) + 0.06 * math.sin(spin * 0.7)) @ rot_z(rad(-6))
        key = []
        # the bow is a flat washer: uniformly filled annulus with a real hole
        na = 5200
        aa = rng.random(na) * TAU
        inner = (0.235 - 0.028) ** 2
        outer = (0.235 + 0.028) ** 2
        rr = np.sqrt(rng.uniform(inner, outer, na))
        bow = np.stack([rr * np.cos(aa), rr * np.sin(aa),
                        rng.normal(0, 0.004, na)], 1).astype(F32)
        key.append(bow + np.array([-0.52, 0.30, 0.0], F32))
        key.append(tube(np.stack([np.array([-0.30, 0.28, 0.0], F32),
                                  np.array([0.72, 0.10, 0.0], F32)]), 16000, 0.016, rng))
        for k, (tx, ty) in enumerate(((0.34, -0.10), (0.52, -0.17))):
            key.append(tube(np.stack([np.array([tx, 0.13 - 0.045 * k, 0.0], F32),
                                      np.array([tx, ty, 0.0], F32)]), 2600, 0.012, rng))
        key = np.concatenate(key).astype(F32) * 1.32
        key = key - np.array([0.06, 0.06, 0.0], F32) @ kR.T
        key = key @ kR.T + K
        # light it from upper-left, so the key reads as a solid lit object
        kn = norm(key - K)
        kc = shade(key, kn, AMBER * 0.30,
                   [((K[0] - 1.4, K[1] + 2.2, K[2] + 1.9), ICE, 2.2),
                    ((K[0] + 2.2, K[1] - 0.6, K[2] + 1.2), AMBER, 1.3)],
                   cam.eye, amb=0.10, rim=0.85, spec=0.9, shin=80)
        S.dots(key, kc, 0.85, blur=0.0)
        # a soft glint inside the bow's hole
        S.dots(sphere_shell(1400, 0.028, 0.0, rng)
               + np.array([0.62 - 0.52 * 1.32 * 0.86, 0.12 + 0.30 * 1.32 * 0.86, 0.55], F32),
               ICE, 0.35, blur=1.6)
        # a slow wave of light crossing the wall
        wv = sstep(2.6, 4.4, v)
        if wv > 0:
            band = np.abs(wall[:, 1] - (1.4 - 2.8 * wv)) < 0.10
            S.dots(wall[band], AMBER, 0.55 * (1 - abs(wv - 0.5) * 1.4), blur=1.0)
    else:
        # ---- optimisation: a cost landscape with a descending probe
        P = floor_pts(-2.8, 2.8, -2.3, 2.1, 170, 140, 0.0)
        X, Z = P[:, 0], P[:, 2]
        h = (0.62 * np.exp(-((X - 1.05) ** 2 + (Z + 0.35) ** 2) * 1.7)
             + 0.92 * np.exp(-((X + 0.85) ** 2 + (Z - 0.65) ** 2) * 1.25)
             + 0.20 * np.sin(X * 2.4) * np.cos(Z * 2.0))
        P[:, 1] = h - 0.70
        col = mix(TEAL * 0.8, AMBER, clamp(h / 1.25))
        S.dots(P, col, 0.20 + 0.50 * clamp(h), blur=1.05)
        # probe: descends the gradient and leaves a bright trail
        q = ease_io(clamp(v / 4.6))
        px = lerp(-0.2, 0.95, q); pz = lerp(-0.5, -0.35, q)
        ph = (0.62 * math.exp(-((px - 1.05) ** 2 + (pz + 0.35) ** 2) * 1.7)
              + 0.92 * math.exp(-((px + 0.85) ** 2 + (pz - 0.65) ** 2) * 1.25))
        ppos = np.array([px, ph - 0.55, pz], F32)
        S.dots(sphere_shell(22000, 0.06, 0.0, rng) + ppos, WHITE, 2.0, blur=0.9)
        tr = np.linspace(0, 1, 900)
        tp = np.stack([lerp(-0.2, px, tr), np.full(900, ph - 0.55),
                       lerp(-0.5, pz, tr)], 1)
        S.dots(tp.astype(F32), AMBER, 0.45, blur=1.3)
    dust_layer(S, t, 130, 2.7, energy=0.15)
    return S


def ov_apps(f, u, t, dur):
    a = fade(u, 0.5, 1.6, 5.0, 5.8)
    label(f, "09 / APPLICATIONS", 120, 96, a * 0.85, 18)
    put_mask(f, text_mask("materials.", "grot", 86, 500, -0.02),
             120, 480, WHITE, a, glow=0.45, reveal=(clamp((u - 0.5) / 0.7), 80))
    put_mask(f, text_mask("new molecules, designed not discovered", "mono", 20, 400, 0.05),
             124, 570, CYAN * 0.85, a * 0.8, glow=0.15)

    b = fade(u, 6.2, 7.2, 10.7, 11.5)
    put_mask(f, text_mask("cryptography.", "grot", 86, 500, -0.02),
             120, 480, WHITE, b, glow=0.45, reveal=(clamp((u - 6.2) / 0.7), 80))
    put_mask(f, text_mask("a key that cannot be copied", "mono", 20, 400, 0.05),
             124, 570, AMBER * 0.9, b * 0.8, glow=0.15)

    c = fade(u, 11.9, 12.9, 16.2, 17.0)
    put_mask(f, text_mask("optimisation.", "grot", 86, 500, -0.02),
             120, 480, WHITE, c, glow=0.45, reveal=(clamp((u - 11.9) / 0.7), 80))
    put_mask(f, text_mask("find the lowest point in a vast landscape", "mono", 20, 400, 0.05),
             124, 570, CYAN * 0.85, c * 0.8, glow=0.15)


# ==================================================================== 10. SHADOW
def sc_shadow(u, t):
    cam = Cam(W, H,
              eye=(kf(u, [(0, 0.05), (19, 0.40)]), kf(u, [(0, 1.68), (19, 1.38)]),
                   kf(u, [(0, 4.75), (19, 4.35)])),
              target=(kf(u, [(0, 0.20), (19, 0.45)]), kf(u, [(0, -0.62), (19, -0.52)]), 0.0),
              fov=44, roll=rad(1.2) * math.sin(u * 0.3))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.66, gy=0.36, gcol=AMBER * 0.40 + TEAL * 0.60, gsize=0.34, gint=0.18)
    rng = jitter(10)

    C = np.array([0.62, 0.48, 0.0], F32)
    rs = 0.60
    L = np.array([-1.20, 3.60, 1.40], F32)
    yfloor = -1.05

    s = (L[1] - yfloor) / max(L[1] - C[1], 1e-3)
    sc = L + s * (C - L)
    srad = rs * s * 0.58

    # dotted floor with a penumbra hole where the sphere's shadow falls
    P = floor_pts(-4.4, 4.4, -4.2, 2.8, 250, 210, yfloor)
    dx = P[:, 0] - sc[0]; dz = P[:, 2] - sc[2]
    dist = np.sqrt(dx * dx + dz * dz)
    pen = np.clip((dist - srad * 0.80) / (srad * 0.75), 0.0, 1.0)
    keep = pen > 0.02
    P = P[keep]; pen = pen[keep]; dist = dist[keep]
    d = cam.depth(P)
    e = 0.46 * np.clip(1.0 - (d - 1.0) / 9.5, 0.05, 1.0)
    edge = (np.clip(1.0 - (np.abs(P[:, 0]) / 4.4) ** 2, 0, 1)
            * np.clip(1.0 - ((P[:, 2] + 0.7) / 3.5) ** 2, 0, 1))
    e = e * (0.10 + 0.90 * pen) * edge ** 0.7
    S.dots(P, mix(TEAL * 0.55, ICE * 0.9, 0.30 * pen), e, blur=1.1)

    # the shadow's rim catches a little light
    th = np.linspace(0, TAU, 1400)
    rim = np.stack([sc[0] + srad * 0.92 * np.cos(th), np.full(1400, yfloor + 0.005),
                    sc[2] + srad * 0.92 * np.sin(th)], 1).astype(F32)
    S.dots(rim, AMBER, 0.30, blur=1.0)

    # light shaft from the lamp
    shaft = np.linspace(L, C, 220)
    S.dots(shaft, AMBER * 0.7, 0.06, blur=2.4)
    S.dots(sphere_shell(1600, 0.05, 0.0, rng) + L, WHITE, 1.2, blur=1.2)

    # the quantum object
    Pq = sphere_shell(120000, rs, 0.015, rng) + C
    Nq = norm(Pq - C)
    cq = shade(Pq, Nq, CYAN * 0.18,
               [(L, AMBER, 2.6), ((-1.4, 1.2, 2.8), CYAN, 0.7)],
               cam.eye, amb=0.08, rim=1.0, spec=1.0, shin=90)
    S.dots(Pq, cq, 0.55 * face_energy(Nq, cam.eye, C, back=0.08), blur=0.35)
    S.dots(sphere_volume(5000, rs * 1.5, rng) + C, CYAN * 0.6, 0.06, blur=3.2)

    # second beat: the object pours down into its shadow and becomes a chip
    q = ease_io(clamp((u - 8.6) / 6.6))
    if q > 0.01:
        nx, nz = 26, 18
        chip = floor_pts(sc[0] - srad * 0.92, sc[0] + srad * 0.92,
                         sc[2] - srad * 0.64, sc[2] + srad * 0.64, nx, nz, yfloor + 0.012)
        pick = rng.integers(0, len(Pq), len(chip))
        Pc = Pq[pick] * (1 - q) + chip * q
        S.dots(Pc, mix(CYAN, AMBER, q), 0.55 + 0.45 * q, blur=0.0)
        pins = np.concatenate([
            np.stack([np.linspace(sc[0] - srad * 1.06, sc[0] + srad * 1.06, 500),
                      np.full(500, yfloor + 0.012), np.full(500, sc[2] - srad * 0.76)], 1),
            np.stack([np.linspace(sc[0] - srad * 1.06, sc[0] + srad * 1.06, 500),
                      np.full(500, yfloor + 0.012), np.full(500, sc[2] + srad * 0.76)], 1)], 0)
        S.dots(pins.astype(F32), AMBER, 0.45 * q, blur=0.0)
    dust_layer(S, t, 140, 2.8, energy=0.15)
    return S


def ov_shadow(f, u, t, dur):
    a = fade(u, 0.6, 1.8, 8.0, 9.2)
    label(f, "10 / THE SHADOW", 120, 96, a * 0.85, 18)
    put_mask(f, text_mask("the classical world", "grot", 72, 300, -0.02),
             120, 440, WHITE, a, glow=0.4, reveal=(clamp((u - 0.6) / 0.8), 80))
    put_mask(f, text_mask("is a shadow.", "grot", 96, 600, -0.03),
             120, 520, WHITE, fade(u, 2.6, 3.4, 8.0, 9.2), glow=0.55,
             reveal=(clamp((u - 2.6) / 0.7), 90))
    put_mask(f, text_mask("of the quantum world", "mono", 20, 400, 0.10),
             124, 646, CYAN * 0.9, fade(u, 3.6, 4.4, 8.0, 9.2) * 0.85, glow=0.15)
    b = fade(u, 10.0, 11.2, 18.0, 19.0)
    put_mask(f, text_mask("machines that operate", "grot", 64, 300, -0.015),
             120, 470, WHITE, b, glow=0.35, reveal=(clamp((u - 10.0) / 0.8), 80))
    put_mask(f, text_mask("the way nature does.", "grot", 64, 500, -0.015),
             120, 545, AMBER, fade(u, 12.0, 12.8, 18.0, 19.0), glow=0.45,
             reveal=(clamp((u - 12.0) / 0.8), 80))


# ==================================================================== 11. OUTRO
_PT_CACHE = {}


def title_points():
    if "P" not in _PT_CACHE:
        rng = np.random.default_rng(31)
        P = text_points("QUANTUM", "grot", 340, 500, tracking=0.01, n=64000, rng=rng)
        # width 5.46 at 1.16 => 5.0 world units, inside the 5.94 frustum
        P = P * 1.06
        P[:, 1] += 0.02
        P = P - np.array([P[:, 0].mean(), P[:, 1].mean(), 0.0], F32)
        _PT_CACHE["P"] = P.astype(F32)
    return _PT_CACHE["P"]


def sc_outro(u, t):
    cam = Cam(W, H, eye=(0.0, 0.0, kf(u, [(0, 5.25), (11.5, 4.85)])),
              target=(0.0, 0.0, 0.0), fov=38, roll=rad(0.6) * math.sin(u * 0.4))
    S = Scene(W, H, cam)
    stage(S, t, gx=0.50, gy=0.50, gcol=CYAN, gsize=0.44, gint=0.26)
    rng = jitter(11)
    Pt = title_points()
    n = len(Pt)
    start = sphere_shell(n, 2.05, 0.30, rng)

    # final collapse: the word becomes one point
    collapse = sstep(8.6, 9.6, u)
    q = ease_io(clamp((u - 0.2) / 3.3))
    stag = rng.random(n) * 0.35
    qq = clamp((q - stag) / (1 - stag))[:, None]
    P = start * (1 - qq) + Pt * qq
    P = P + np.stack([np.sin(u * 0.7 + Pt[:, 0] * 2) * 0.010,
                      np.sin(u * 0.9 + Pt[:, 1] * 2) * 0.010,
                      np.zeros(n, F32)], 1) * q
    vgrad = clamp(0.5 + 0.85 * (Pt[:, 1] - Pt[:, 1].min())
                  / max(Pt[:, 1].max() - Pt[:, 1].min(), 1e-3))
    col = mix(CYAN, ICE, vgrad * 0.75)
    e = (0.32 + 0.30 * q) * (0.88 + 0.12 * math.sin(u * 1.6))
    if collapse > 0:
        # shrink toward the origin but never stack every point on one pixel
        P = P * (1.0 - 0.945 * collapse)
        e = e * (1.0 - 0.72 * collapse)
    S.dots(P, col, e, blur=0.95)
    if collapse > 0.2:
        th = np.linspace(0, TAU, 900)
        rr = 0.25 + 2.4 * collapse
        ring = np.stack([rr * np.cos(th), rr * np.sin(th), np.zeros(900)], 1).astype(F32)
        S.dots(ring, CYAN, 0.30 * (1 - collapse), blur=2.2)
        S.dots(fib_sphere(3000, 0.030 + 0.022 * collapse, 0.002, rng), ICE, 1.05, blur=0.0)
    dust_layer(S, t, 160, 3.0, energy=0.16)
    return S


def ov_outro(f, u, t, dur):
    a = fade(u, 3.6, 4.6, 10.2, 11.2)
    put_mask(f, text_mask("a new language.", "serif_i", 54, None),
             120, 800, ICE, a * 0.95, glow=0.35)
    label(f, "11 / QUANTUM", 120, 96, fade(u, 4.0, 5.0, 10.2, 11.2) * 0.8, 18)
    b = fade(u, 6.0, 7.0, 10.4, 11.4)
    put_mask(f, text_mask("how it works", "mono", 18, 400, 0.22),
             1590, 800, CYAN * 0.8, b * 0.7, glow=0.12)


# =================================================================== assembly
SCENES = [sc_hook, sc_bits, sc_useit, sc_qubit, sc_explode, sc_entangle,
          sc_interference, sc_decohere, sc_apps, sc_shadow, sc_outro]
OVERLAYS = [ov_hook, ov_bits, ov_useit, ov_qubit, ov_explode, ov_entangle,
            ov_interf, ov_decohere, ov_apps, ov_shadow, ov_outro]

# cuts that get a light flash (section index -> intensity)
FLASH = {2: 0.35, 4: 0.55, 8: 0.30, 10: 0.55}


def build(t):
    idx, u, u0, dur = locate(t)
    return SCENES[idx](u, t)


def render_scene(t, nsub=NSUB):
    acc = None
    samples = shutter_samples(t, FPS, nsub)
    for tt in samples:
        S = build(tt)
        img = S.compose()
        acc = img if acc is None else acc + img
    return acc / max(len(samples), 1)


def render_frame(t):
    img = render_scene(t, NSUB)
    frame = finish(img, OUT_W, OUT_H, exposure=0.98, grain=0.014, vig=0.5, ca=0.0014, t=t)
    f = frame.astype(F32) / 255.0

    # cinematic text scrim: the left third sits a stop down
    idx0 = locate(t)[0]
    if idx0 != 10:
        xx = np.linspace(0.0, 1.0, OUT_W, dtype=F32)[None, :]
        scrim = 1.0 - 0.42 * np.clip((0.50 - xx) / 0.50, 0.0, 1.0) ** 1.15
        f = f * scrim[..., None]

    idx, u, u0, dur = locate(t)
    OVERLAYS[idx](f, u, t, dur)
    if idx in FLASH:
        k = math.exp(-u * 26.0)
        f = f + FLASH[idx] * k
    f = np.clip(f, 0, 1)
    return (f * 255.0 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    import time, sys
    t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    st = time.time()
    fr = render_frame(t0)
    print("render", round(time.time() - st, 2), "s", fr.shape)
