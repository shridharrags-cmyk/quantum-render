"""
film.py — QUANTUM, a 192 s cinematic explainer.

Every pixel is a point of light in a 3-D scene: the reference videos' dotted
visual language, rendered with perspective, lighting, defocus, motion blur,
bloom, anamorphic streaks and film grain.

Sections follow the narration (audio/timeline.json), so every cut lands on a
sentence boundary.
"""
import math
import os
import numpy as np
from qfx import (Cam, Scene, F32, TAU, rad, clamp, smoothstep, sstep, ease_out,
                 ease_in, ease_io, lerp, mix, norm, rot_x, rot_y, rot_z,
                 sphere_shell, sphere_volume, torus, disc, grid3, bez, tube, wire_sphere,
                 shade, blur_rgb, resize_f, bloom, grade, aces, chromatic,
                 add_grain, finish, vignette, text_mask, put_mask, font,
                 label, rule, corner_marks, text_points, shutter_samples,
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
AMBER  = np.array([1.00, 0.58, 0.18], F32)
MINT   = np.array([0.55, 1.00, 0.82], F32)
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
            if isinstance(va, (tuple, list, np.ndarray)):
                return tuple(float(x + (y - x) * p) for x, y in zip(va, vb))
            return va + (vb - va) * p
    return keys[-1][1]


def screen_glow(S, cx, cy, r, color, inten=1.0, power=2.4):
    H, W = S.H, S.W
    yy, xx = np.mgrid[0:H, 0:W].astype(F32)
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
    g = np.exp(-d2 * power)[..., None]
    S._buf(0.0)[...] += g * np.asarray(color, F32)[None, None, :] * inten


def dust_layer(S, t, n=260, blur=2.4, box=(9, 4, 9), energy=0.30, col=None):
    rng = np.random.default_rng(99)
    P = dust(n, rng, box=box, t=t, speed=0.03)
    d = S.cam.depth(P)
    e = energy * np.clip(1.35 - d / 9.0, 0.05, 1.0)
    S.dots(P, col if col is not None else ICE * 0.8, e, blur=blur)


def ring3d(n, R=1.0, r=0.02, tilt=0.0, yaw=0.0, rng=None):
    P = torus(n, R, r, rng)
    Rm = rot_y(yaw) @ rot_x(tilt)
    return P @ Rm.T


def blink_energy(u, phases, freq=1.7, duty=0.45):
    v = np.sin(u * freq * TAU + phases)
    return np.where(v > math.cos(duty * math.pi), 1.0, 0.18).astype(F32)


# ================================================================== 01. HOOK
def sc_hook(u, t):
    cam = Cam(W, H, eye=(0.0, 0.35, kf(u, [(0, 2.0), (5.0, 3.5), (11.0, 5.0)])),
              target=(0.0, 0.05, 0.0), fov=40, roll=0.02 * math.sin(u * 0.3))
    S = Scene(W, H, cam)
    rng = jitter(1)
    screen_glow(S, W * 0.5, H * 0.52, W * 0.42, INDIGO * 0.5 + CYAN * 0.15, 0.10)
    dot_ground(cam, S, t, n=6300, y=-1.35, extent=11, col=TEAL * 0.7, energy=0.22, blur=1.6)

    # the point of light -> a shell of 160k
    birth = sstep(1.2, 3.4, u)
    n = 70000 + int(90000 * birth)
    P0 = sphere_shell(n, 0.55, 0.012, rng)
    spin = u * 0.35
    P = P0 @ rot_y(spin).T + np.array([0.62, 0.02, 0.0], F32)
    # slow breathing
    P = P * (1.0 + 0.03 * math.sin(u * 1.4))
    # detonation: particles start collapsed and fly out
    q = ease_out(clamp((u - 1.2) / 2.6), 3.0)
    P = P * (0.02 + 0.98 * q)
    N = norm(P)
    lights = [((2.6, 2.4, 3.2), ICE, 2.4), ((-2.4, -1.0, 1.8), AMBER, 0.7)]
    col = shade(P, N, CYAN * 0.26 + TEAL * 0.10, lights, cam.eye, amb=0.035,
                rim=0.55, spec=0.45, fog=VOID * 2.0, fog_density=0.02)
    S.dots(P, col, 0.42, blur=0.9)
    # bright core that collapses
    core = sphere_volume(int(3000 * (1 - sstep(2.8, 4.2, u))) + 1, 0.06, rng)
    S.dots(core, ICE, 1.4, blur=2.2)
    S.dots(core, WHITE, 0.9, blur=0.8)
    # orbital ring
    ring = ring3d(24000, 0.92, 0.012, tilt=rad(18), yaw=u * 0.6, rng=rng) + np.array([0.62, 0, 0], F32)
    rc = shade(ring, norm(ring), TEAL, [((0, 2, 2), ICE, 1.4)], cam.eye, amb=0.2, rim=1.0)
    S.dots(ring, rc, 0.42, blur=0.9)
    # second ring
    ring2 = ring3d(14000, 1.18, 0.010, tilt=rad(-24), yaw=-u * 0.4, rng=rng) + np.array([0.62, 0, 0], F32)
    S.dots(ring2, TEAL * 0.85, 0.22, blur=1.5)
    # unseen machinery: a vast shell of far stars
    stars = sphere_shell(9000, 8.0, 0.35, rng)
    S.dots(stars, ICE * 0.5, 0.12, blur=1.4)
    dust_layer(S, t, 200, 2.8)
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


# ================================================================== 02. BITS
def sc_bits(u, t):
    fwd = kf(u, [(0, 0.0), (13, 6.0), (20.5, 10.5)])
    cam = Cam(W, H, eye=(kf(u, [(0, 0.35), (13, -0.15), (20.5, 0.0)]),
                          kf(u, [(0, 1.15), (13, 0.85), (20.5, 0.62)]),
                          kf(u, [(0, 3.2), (13, 1.4), (20.5, 0.9)])),
              target=(0.0, kf(u, [(0, 0.05), (13, 0.15), (20.5, 0.30)]), 0.0 - fwd),
              fov=52, roll=rad(0.8) * math.sin(u * 0.35))
    S = Scene(W, H, cam)
    rng = jitter(2)
    screen_glow(S, W * 0.5, H * 0.40, W * 0.55, TEAL * 0.55, 0.14)

    # ---- the bit field: thousands of switches, each blinking on its own clock
    gx = np.arange(-14.0, 14.0, 0.30)
    gz = np.arange(-23.0, 1.6, 0.30)
    X, Z = np.meshgrid(gx, gz)
    P = np.stack([X.ravel(), np.full(X.size, 0.0, F32), Z.ravel()], 1)
    ph = rng.random(len(P)) * TAU
    rate = 0.7 + 1.5 * rng.random(len(P))
    on = (np.sin(u * rate * TAU + ph) > 0.15).astype(F32)
    on = 0.10 + 0.90 * on * (0.55 + 0.45 * on)
    rare = rng.random(len(P)) < 0.035
    col = np.where(rare[:, None], AMBER, np.where((rng.random(len(P)) < 0.15)[:, None], ICE, CYAN))
    e = on * np.clip(1.85 - 0.045 * cam.depth(P), 0.25, 1.85)
    S.dots(P, col, e * 0.85, blur=1.25)

    # perspective rails: two brighter lines of dots running to the horizon
    for sx in (-3.2, 3.2):
        rail = np.stack([np.full(900, sx, F32), np.full(900, 0.02, F32),
                         np.linspace(1.0, -22, 900)], 1)
        S.dots(rail, ICE, 0.55, blur=1.6)

    # ---- a wall of states at the far edge
    wx = np.arange(-13, 13, 0.26)
    wy = np.arange(0.0, 8.0, 0.26)
    WX, WY = np.meshgrid(wx, wy)
    WP = np.stack([WX.ravel(), WY.ravel(), np.full(WX.size, -21.5, F32)], 1)
    won = (np.sin(u * 1.6 + rng.random(len(WP)) * TAU) > 0.35).astype(F32)
    S.dots(WP, TEAL * 0.95, (0.16 + 0.75 * won) * 0.7, blur=1.4)

    # ---- dive into one cell: the atom
    atom_a = sstep(17.0, 20.0, u)
    if atom_a > 0:
        ap = np.array([0.0, 0.55, kf(u, [(17, -8.2), (21, -7.2)])], F32)
        c = sphere_volume(int(14000 * atom_a) + 1, 0.075, rng) + ap
        S.dots(c, WHITE, 1.7 * atom_a, blur=1.5)
        sh1 = sphere_shell(int(42000 * atom_a) + 1, 0.62, 0.02, rng)
        j = 1.0 + 0.03 * np.sin(u * 8.0 + np.arange(len(sh1)) * 0.013)[:, None]
        sh1 = sh1 * j + ap
        S.dots(sh1, CYAN * 0.95, 0.55 * atom_a, blur=1.0)
        sh2 = sphere_shell(int(30000 * atom_a) + 1, 1.02, 0.05, rng) + ap
        S.dots(sh2, TEAL * 0.8, 0.32 * atom_a, blur=1.3)
        # the wall recedes behind it, trembling
        S.dots(WP, AMBER * 0.7, 0.35 * atom_a, blur=1.6)
    dust_layer(S, t, 190, 2.8, energy=0.22)
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


# ============================================================== 03. USE IT
def sc_useit(u, t):
    cam = Cam(W, H, eye=(kf(u, [(0, 0.0), (3.6, -1.4)]), 0.25, kf(u, [(0, 3.2), (3.6, 2.4)])),
              target=(0.0, 0.0, 0.0), fov=42, roll=rad(-2) * u / 3.6)
    S = Scene(W, H, cam)
    rng = jitter(3)
    screen_glow(S, W * 0.5, H * 0.5, W * 0.3, CYAN * 0.6, 0.16)
    # atom core condenses then explodes into an ordered orbit
    a = sstep(0.0, 0.7, u)
    c = sphere_volume(9000, 0.05, rng)
    S.dots(c, WHITE, 1.5, blur=1.6)
    burst = ease_out(clamp((u - 0.35) / 1.4), 2.6)
    n = 90000
    P = sphere_shell(n, 0.35, 0.02, rng)
    P = P * (1.0 + 2.2 * burst)
    P = P @ rot_x(rad(62)).T @ rot_z(u * 0.9).T
    S.dots(P, mix(CYAN, AMBER, 0.25 * burst), 0.5 * (1 - 0.55 * burst), blur=1.1)
    # a clean bright orbit ring
    R = ring3d(50000, 1.25, 0.012, tilt=rad(66), yaw=u * 1.5, rng=rng)
    S.dots(R, ICE, 0.65, blur=0.9)
    # shockwave
    sw = clamp(u / 1.8)
    if sw < 1.0:
        w = 0.55 + sw * 3.2
        swP = ring3d(26000, w, 0.01, tilt=rad(70), yaw=0.3, rng=rng)
        S.dots(swP, mix(WHITE, TEAL, sw), 0.9 * (1 - sw) ** 2, blur=2.4)
    dust_layer(S, t, 180, 2.6)
    return S


def ov_useit(f, u, t, dur):
    a = fade(u, 0.15, 0.5, 3.1, 3.6)
    put_mask(f, text_mask("use it.", "grot", 150, 700, -0.03),
             120, 430, WHITE, a, glow=0.7, reveal=(clamp((u - 0.15) / 0.5), 100))
    label(f, "03 / THE TURN", 120, 96, a * 0.85, 18)
    l2 = fade(u, 1.4, 2.2, 3.1, 3.6)
    put_mask(f, text_mask("stop fighting the strangeness", "mono", 21, 400, 0.09),
             124, 620, CYAN, l2 * 0.8, glow=0.2)


# ================================================================= 04. QUBIT
def spinning_coin(n, R, thick, rng):
    """A coin as a point cloud, shaded front/back/rim. Returns P, N."""
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
    ex = kf(u, [(0, 0.95), (8, -0.75), (14, 0.10), (23, 0.35)])
    ey = kf(u, [(0, 0.30), (8, 0.50), (23, 0.16)])
    ez = kf(u, [(0, 4.25), (8, 3.80), (14, 4.10), (23, 3.65)])
    cam = Cam(W, H, eye=(ex, ey, ez), target=(kf(u, [(0, 0.30), (8, -0.10), (23, 0.18)]), 0.03, 0.0),
              fov=40, roll=rad(1.1) * math.sin(u * 0.22))
    S = Scene(W, H, cam)
    rng = jitter(4)
    screen_glow(S, W * 0.60, H * 0.46, W * 0.30, CYAN * 0.45, 0.15)
    dot_ground(cam, S, t, n=6400, y=-1.30, extent=9, col=TEAL * 0.6, energy=0.30, blur=1.7)
    dust_layer(S, t, 240, 2.9, energy=0.28)

    base = np.array([kf(u, [(0, 0.0), (8, 0.30), (14, 0.55), (23, 0.35)]),
                     kf(u, [(0, 0.18), (23, 0.02)]), 0.0], F32)
    sphere_a = 1.0 - sstep(13.4, 14.6, u)

    if sphere_a > 0:
        # faint volume so the globe has body
        fill = sphere_shell(42000, 0.78, 0.03, rng) + base
        fc = shade(fill, norm(fill - base), CYAN * 0.16 + TEAL * 0.06,
                   [((2.2, 2.4, 3.0), ICE, 1.6)], cam.eye, amb=0.02, rim=0.25)
        S.dots(fill, fc, 0.30 * sphere_a, blur=1.0)
        # the wireframe globe: latitude + meridians
        wire = wire_sphere(9, 16, 0.78, per=520) + base
        S.dots(wire, CYAN, 0.62 * sphere_a, blur=0.95)
        # bright equator
        eq = ring3d(7000, 0.785, 0.004, tilt=0, yaw=0, rng=rng) + base
        S.dots(eq, ICE, 0.85 * sphere_a, blur=0.9)
        # poles
        for sy in (-1, 1):
            S.dots(sphere_volume(900, 0.018, rng) + base + np.array([0, 0.785 * sy, 0], F32),
                   WHITE, 1.5 * sphere_a, blur=1.8)

        # state vector: one arrow, then a fan of every possibility
        split = sstep(7.4, 9.2, u)
        vec_a = sphere_a * (1 - 0.0 * split)
        ang0 = rad(62)
        wob = 0.10 * math.sin(u * 0.9)
        th = ang0 + wob
        if split > 0.01:
            m = 90
            ths = np.linspace(rad(18), rad(162), m)
            for j in range(m):
                q = ths[j]
                tip = base + np.array([0.78 * math.sin(q), 0.78 * math.cos(q), 0.0], F32)
                seg = np.linspace(0, 1, 40)[:, None]
                line = base + (tip - base) * seg
                w = 1.0 if (j % 12 == 0) else 0.35
                S.dots(line, mix(AMBER, CYAN, j / m), 0.30 * split * w, blur=1.2)
        tip = base + np.array([0.78 * math.sin(th), 0.78 * math.cos(th), 0.0], F32)
        seg = np.linspace(0, 1, 700)[:, None]
        vec = base + (tip - base) * seg
        S.dots(vec, mix(WHITE, AMBER, 0.25 * split), 0.95 * vec_a, blur=1.0)
        S.dots(sphere_volume(1600, 0.035, rng) + tip, WHITE, 1.4 * vec_a, blur=1.8)
        # arrow head
        ah = sphere_volume(700, 0.022, rng) + tip
        S.dots(ah, AMBER, 1.0 * vec_a, blur=1.4)
        # spin precession ring around the arrow
        prec = ring3d(9000, 0.83, 0.006, tilt=rad(70), yaw=u * 1.3, rng=rng) + base
        S.dots(prec, TEAL, 0.30 * sphere_a, blur=1.2)

    # ---- the coin, second half
    coin_a = sstep(14.0, 15.2, u) * (1 - sstep(19.4, 20.2, u))
    if coin_a > 0:
        ang = (u - 14.0) * 3.1
        settle = sstep(18.6, 19.6, u)
        ang = ang * (1 - settle * 0.85)
        Pc, Nc = spinning_coin(150000, 0.72, 0.045, rng)
        Rm = rot_y(ang * 0.55) @ rot_x(ang)
        Pc = Pc @ Rm.T + np.array([0.35, 0.05, 0], F32)
        Nc = Nc @ Rm.T
        lights = [((2.2, 2.6, 3.2), ICE, 2.1), ((-2.4, -1.2, 1.4), AMBER, 0.55)]
        cc = shade(Pc, Nc, CYAN * 0.28 + WHITE * 0.06, lights, cam.eye, amb=0.03,
                   rim=0.55, spec=0.9, shin=70)
        r = np.linalg.norm(Pc[:, :2], axis=1)
        eng = np.exp(-((r - 0.52) / 0.018) ** 2)[:, None]
        cc = cc + eng * ICE * 0.55
        S.dots(Pc, cc, 0.48 * coin_a, blur=0.9)
        fl = sstep(19.25, 19.35, u) * (1 - sstep(19.35, 20.2, u))
        if fl > 0:
            screen_glow(S, W * 0.5, H * 0.5, W * 0.55, ICE, 0.5 * fl)
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

    c = fade(u, 15.4, 16.4, 18.4, 19.0)
    put_mask(f, text_mask("a blur of possibility", "serif_i", 62, None),
             120, 520, ICE, c, glow=0.3)

    d = fade(u, 19.7, 20.4, 22.4, 23.0)
    put_mask(f, text_mask("α|0⟩ + β|1⟩", "serif", 108, None, 0.01),
             118, 470, AMBER, d, glow=0.55, reveal=(clamp((u - 19.7) / 0.5), 70))
    put_mask(f, text_mask("until you look", "mono", 22, 400, 0.10),
             124, 620, ICE * 0.9, d * 0.8, glow=0.18)


# ============================================================= 05. EXPLOSION
def sc_explode(u, t):
    pull = kf(u, [(0, 3.1), (5.5, 3.5), (8.2, 5.0), (9.9, 6.6)])
    cam = Cam(W, H, eye=(kf(u, [(0, 0.35), (9.9, -0.5)]),
                          kf(u, [(0, 0.55), (9.9, 1.05)]), pull),
              target=(0.0, 0.0, 0.0), fov=44, roll=rad(2.5) * math.sin(u * 0.4))
    S = Scene(W, H, cam)
    rng = jitter(5)
    screen_glow(S, W * 0.5, H * 0.46, W * 0.40, INDIGO * 0.55 + TEAL * 0.25, 0.14)

    k = clamp((u - 0.25) / 6.4)
    n = int(min(2 ** (1 + k * 16.0), 190000))
    n = max(n, 2)
    P = rng.uniform(-1.0, 1.0, (n, 3)).astype(F32)
    # the counting cage: 12 edges, always visible, gives the growth a scale
    E = []
    for a in (-1, 1):
        for b in (-1, 1):
            E.append(np.stack([np.full(240, a), np.full(240, b), np.linspace(-1, 1, 240)], 1))
            E.append(np.stack([np.full(240, a), np.linspace(-1, 1, 240), np.full(240, b)], 1))
            E.append(np.stack([np.linspace(-1, 1, 240), np.full(240, a), np.full(240, b)], 1))
    E = np.concatenate(E).astype(F32)
    S.dots(E, TEAL, 0.30, blur=1.2)

    # burst: the cube's contents blow outward into a sphere
    burst = sstep(6.9, 8.8, u)
    R = np.linalg.norm(P, axis=1, keepdims=True)
    P = P * (1.0 + burst * (1.6 * (0.35 + 0.65 * R)))
    P = P @ rot_y(u * 0.30).T
    d = np.linalg.norm(P, axis=1)
    col = mix(CYAN, AMBER, clamp(d / 2.2) * 0.30)
    hot = np.exp(-d * d * 0.5)[:, None]
    e = 0.42 + 0.55 * hot + 0.10 * np.sin(u * 5.0)
    S.dots(P, col, e * 0.85, blur=1.0)
    # radial filaments once exploded
    if burst > 0.02:
        for j in range(26):
            th = j * TAU / 26 + 0.2
            ph = math.sin(j * 1.7) * 0.9
            dirv = np.array([math.cos(th) * math.cos(ph), math.sin(ph), math.sin(th) * math.cos(ph)], F32)
            ln = np.linspace(0.2, 2.6, 380)
            fil = dirv[None, :] * ln[:, None] + rng.normal(0, 0.012, (380, 3))
            S.dots(fil, mix(CYAN, WHITE, 0.3), 0.55 * burst, blur=1.2)
    S.dots(sphere_volume(9000, 0.05, rng), WHITE, 1.5, blur=2.2)
    dust_layer(S, t, 200, 2.9)
    return S

def ov_explode(f, u, t, dur):
    a = fade(u, 0.2, 0.8, 8.8, 9.6)
    label(f, "05 / SUPERPOSITION", 120, 96, a * 0.85, 18)
    # live counter 2^n
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


# =========================================================== 06. ENTANGLEMENT
def sc_entangle(u, t):
    ex = kf(u, [(0, 0.2), (10, 2.3), (16, -1.2), (23, 0.1)])
    ez = kf(u, [(0, 5.2), (10, 4.2), (23, 3.7)])
    ey = kf(u, [(0, 0.25), (10, 0.85), (23, 0.2)])
    cam = Cam(W, H, eye=(ex, ey, ez), target=(0.0, 0.0, 0.0), fov=40,
              roll=rad(2.0) * math.sin(u * 0.35))
    S = Scene(W, H, cam)
    rng = jitter(6)
    screen_glow(S, W * 0.5, H * 0.48, W * 0.36, INDIGO * 0.4 + CYAN * 0.2, 0.12)
    dot_ground(cam, S, t, n=5400, y=-1.25, extent=9, col=TEAL * 0.55, energy=0.18, blur=1.8)

    x = 1.55
    A = np.array([-x, 0.0, 0.0], F32)
    B = np.array([x, 0.0, 0.0], F32)
    nA = 78000
    PA = sphere_shell(nA, 0.52, 0.02, rng) + A
    PB = sphere_shell(nA, 0.52, 0.02, rng) + B
    lights = [((0.0, 2.6, 3.0), ICE, 1.2), ((-3.0, -1.0, 1.0), AMBER, 0.4)]
    cA = shade(PA, norm(PA - A), CYAN * 0.6 + WHITE * 0.1, lights, cam.eye, amb=0.08, rim=1.2, spec=0.5)
    cB = shade(PB, norm(PB - B), INDIGO * 0.55 + CYAN * 0.2, lights, cam.eye, amb=0.08, rim=1.2, spec=0.5)
    S.dots(PA, cA, 0.58, blur=0.95)
    S.dots(PB, cB, 0.58, blur=0.95)

    # braided filament: three helical strands
    tt = np.linspace(0, 1, 1400)
    cat = bez(np.array([-x, 0, 0], F32), np.array([-0.5, 0.75, 0.4], F32),
              np.array([0.5, 0.75, 0.4], F32), np.array([x, 0, 0], F32), tt)
    pulse = 0.55 + 0.45 * np.sin(u * 2.2)
    for k in range(3):
        ph = k * TAU / 3
        wig = 0.055 * np.stack([np.zeros_like(tt), np.sin(tt * 26 + ph + u * 2.0),
                                np.cos(tt * 26 + ph + u * 2.0)], 1)
        line = cat + wig
        # densify
        idx = rng.integers(0, len(line), 26000)
        jj = rng.normal(0, 0.008, (26000, 3))
        P = line[idx] + jj
        e = 0.5 * (0.6 + 0.4 * np.sin(u * 5 - tt[idx] * 12))
        S.dots(P, mix(TEAL, ICE, pulse * 0.5), e, blur=1.0)
    # travelling measurement pulses
    for k in range(2):
        q = ((u * 0.55 + k * 0.5) % 1.0)
        if q < 1:
            i0 = int(q * (len(cat) - 1))
            seg = cat[max(i0 - 4, 0):i0 + 4]
            if len(seg) > 0:
                S.dots(seg, WHITE, 1.0, blur=2.6)

    # measurement: one flashes, so does the other, instantly
    m = sstep(18.6, 18.75, u) * (1 - sstep(19.8, 21.0, u))
    if m > 0:
        S.dots(sphere_volume(9000, 0.12, rng) + A, WHITE, 2.4 * m, blur=3.0)
        S.dots(sphere_volume(9000, 0.12, rng) + B, WHITE, 2.4 * m, blur=3.0)
        for c in (A, B):
            sw = clamp((u - 18.6) / 1.2)
            if sw < 1:
                rr = 0.4 + sw * 2.4
                ring = ring3d(16000, rr, 0.012, tilt=rad(72), yaw=0.4, rng=rng) + c
                S.dots(ring, mix(WHITE, CYAN, sw), 0.8 * (1 - sw) ** 2, blur=2.2)
    # subtle link line at rest
    S.dots(cat[rng.integers(0, len(cat), 6000)], TEAL, 0.16, blur=2.0)
    dust_layer(S, t, 210, 2.9)
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


# ============================================================ 07. INTERFERENCE
def sc_interference(u, t):
    cam = Cam(W, H, eye=(kf(u, [(0, -0.25), (24, 0.55)]), kf(u, [(0, 0.62), (12, 0.95), (24, 0.70)]),
                          kf(u, [(0, 2.55), (24, 2.15)])),
              target=(kf(u, [(0, -0.15), (24, 0.25)]), kf(u, [(0, -0.10), (12, 0.05), (24, 0.15)]),
                      kf(u, [(0, -1.9), (24, -1.5)])),
              fov=56, roll=rad(0.8))
    S = Scene(W, H, cam)
    rng = jitter(7)
    screen_glow(S, W * 0.5, H * 0.58, W * 0.50, TEAL * 0.5, 0.13)

    nx, nz = 300, 220
    P = grid3(nx, nz, sx=9.6, sz=7.4, y=0.0)
    P[:, 2] -= 2.2
    X, Z = P[:, 0], P[:, 2]

    k = 3.2
    w = 1.9
    s1 = np.array([-1.25, 0.0, -1.0], F32)
    s2 = np.array([1.25, 0.0, -1.0], F32)
    r1 = np.sqrt((X - s1[0]) ** 2 + (Z - s1[2]) ** 2) + 1e-4
    r2 = np.sqrt((X - s2[0]) ** 2 + (Z - s2[2]) ** 2) + 1e-4
    a1 = np.sin(k * r1 - w * (u + t) * 2.0) / (1 + 0.50 * r1)
    a2 = np.sin(k * r2 - w * (u + t) * 2.0) / (1 + 0.50 * r2)
    amp = a1 + a2

    ans = sstep(13.0, 20.0, u)
    r3 = np.sqrt((X - 0.0) ** 2 + (Z + 0.6) ** 2) + 1e-4
    a3 = np.sin(k * r3 - w * (u + t) * 2.0) * np.exp(-((r3 - 1.0) / 1.6) ** 2)
    amp = amp * (1 - 0.60 * ans) + a3 * ans * 1.6

    col_a = sstep(18.0, 21.0, u)
    P[:, 1] = amp * 0.06 * (1 - col_a)
    energy = (np.abs(amp) ** 1.5) * 1.5 + 0.05
    energy = energy * (0.55 + 0.45 * (1 - col_a))
    col = mix(TEAL, mix(CYAN, AMBER, clamp(np.abs(amp) * 0.6)), 0.35)
    S.dots(P, col, energy, blur=1.05)

    # the two emitters
    for s in (s1, s2):
        S.dots(sphere_volume(2200, 0.055, rng) + s, WHITE, 1.2, blur=2.2)
        S.dots(sphere_volume(900, 0.10, rng) + s, TEAL, 0.35, blur=3.2)
    if ans > 0.02:
        S.dots(sphere_volume(2600, 0.06, rng) + np.array([0, 0.02, -0.6], F32),
               AMBER, 1.4 * ans, blur=2.2)

    if col_a > 0.01:
        n = 70000
        h = 3.8
        yy = rng.random(n) ** 0.75 * h
        ang = rng.random(n) * TAU
        rr = (0.035 + 0.03 * np.exp(-yy)) * (0.3 + 0.7 * rng.random(n))
        CP = np.stack([np.cos(ang) * rr, yy - 0.05, -0.6 + np.sin(ang) * rr], 1)
        e = (1.0 - yy / h) ** 0.75 * (0.75 + 0.25 * math.sin(u * 3))
        S.dots(CP, mix(AMBER, ICE, clamp(yy / h)), e * col_a * 1.0, blur=1.0)
        RP = CP.copy(); RP[:, 1] *= -0.55
        S.dots(RP, TEAL, e * col_a * 0.20, blur=2.8)
    hz = np.stack([np.linspace(-8, 8, 900), np.full(900, 0.01), np.full(900, -6.6)], 1)
    S.dots(hz, TEAL, 0.16, blur=1.8)
    dust_layer(S, t, 240, 3.0, energy=0.24)
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
    cam = Cam(W, H, eye=(kf(u, [(0, 0.1), (24, 1.2)]), kf(u, [(0, 0.55), (24, 0.25)]),
                          kf(u, [(0, 3.5), (24, 3.9)])),
              target=(0.0, 0.0, 0.0), fov=42, roll=rad(1.8) * math.sin(u * 0.3))
    S = Scene(W, H, cam)
    rng = jitter(8)
    screen_glow(S, W * 0.5, H * 0.5, W * 0.28, INDIGO * 0.35 + CYAN * 0.3, 0.16)

    # cryostat: three concentric rings
    for i, (R, tilt, ph) in enumerate([(1.55, 78, 0.0), (1.95, 62, 1.0), (2.35, 88, 2.0)]):
        ring = ring3d(34000, R, 0.014, tilt=rad(tilt) + 0.06 * math.sin(u * 0.4 + ph),
                      yaw=u * 0.18 * (1 if i % 2 == 0 else -1) + ph, rng=rng)
        e = 0.30 + 0.10 * math.sin(u * 0.8 + ph)
        S.dots(ring, TEAL * 0.9, e, blur=1.0)
    # cold fog particles
    fogp = sphere_volume(26000, 2.2, rng)
    fogp[:, 1] *= 0.55
    S.dots(fogp, INDIGO * 0.5, 0.05, blur=3.2)

    # the computing core
    core = sphere_shell(52000, 0.42, 0.012, rng)
    core = core @ rot_y(u * 0.5).T
    core_c = shade(core, norm(core), CYAN * 0.55 + WHITE * 0.15,
                   [((1.2, 2.2, 2.0), ICE, 1.3), ((-1.4, -0.8, 0.6), AMBER, 0.35)],
                   cam.eye, amb=0.10, rim=1.2, spec=0.6)
    S.dots(core, core_c, 0.62, blur=0.95)
    S.dots(sphere_volume(3500, 0.04, rng), WHITE, 1.3, blur=2.2)

    # heat intrudes: amber sparks with trails
    for k in range(7):
        phase = (u * 0.42 + k * 0.137) % 1.0
        # deterministic impact beat
        beat = 3.6 + k * 2.9
        if 0.4 < phase < 0.62:
            q = (phase - 0.4) / 0.22
            start = np.array([2.8 * math.cos(k * 2.1), 1.4 * math.sin(k * 1.7), 1.2], F32)
            end = np.array([0.5 * math.cos(k * 1.3), 0.35 * math.sin(k * 1.1), 0.2], F32)
            p = start * (1 - q) + end * q
            trail = p[None, :] + np.linspace(-1, 0, 200)[:, None] * (start - end)[None, :] * 0.09
            S.dots(trail, AMBER, (0.5 + 0.5 * q) * 1.0, blur=1.4)
            S.dots(p[None, :], WHITE, 1.4, blur=2.0)
            if q > 0.92:
                # impact: scatter a patch of the core
                h = int(u * 3.0) + k
                rr = np.random.default_rng(h)
                patch = core[np.linalg.norm(core - end, axis=1) < 0.30]
                if len(patch):
                    patch = patch + rr.normal(0, 0.028, patch.shape)
                    S.dots(patch, AMBER, 1.1, blur=1.8)
        # aftermath glow
        for kk in range(3):
            imp_t = 3.6 + kk * 2.9
            if 0 < u - imp_t < 1.2:
                q = (u - imp_t) / 1.2
                c = np.array([0.5 * math.cos(kk * 1.3), 0.35 * math.sin(kk * 1.1), 0.2], F32)
                ring = ring3d(9000, 0.15 + q * 1.2, 0.01, tilt=rad(70), yaw=0.5, rng=rng) + c
                S.dots(ring, AMBER, 0.55 * (1 - q) ** 2, blur=2.2)

    # error correction: a lattice of cells that lights up and repairs
    ec = sstep(17.0, 19.0, u)
    if ec > 0:
        gx, gy = 9, 7
        cell = np.zeros((gx * gy, 3), F32)
        cell[:, 0] = (np.arange(gx * gy) % gx - gx / 2) * 0.42
        cell[:, 1] = (np.arange(gx * gy) // gx - gy / 2) * 0.42
        cell[:, 2] = -0.9
        on = (np.sin(u * 2.0 + np.arange(len(cell)) * 0.7) > 0.2).astype(F32) * ec
        S.dots(cell, MINT, 0.35 + 0.5 * on, blur=1.2)
        edges = np.stack([np.minimum(cell[:, 0] % 0.42, 0.42 - cell[:, 0] % 0.42),
                          np.minimum(cell[:, 1] % 0.42, 0.42 - cell[:, 1] % 0.42)], 1)
        S.dots(cell[on > 0.5], MINT, 0.9, blur=1.6)
    dust_layer(S, t, 180, 3.0, energy=0.20)
    return S


def ov_decohere(f, u, t, dur):
    a = fade(u, 0.5, 1.5, 8.0, 9.0)
    label(f, "08 / DECOHERENCE", 120, 96, a * 0.85, 18)
    put_mask(f, text_mask("the moment you look", "grot", 78, 300, -0.02),
             120, 452, WHITE, a, glow=0.4, reveal=(clamp((u - 0.5) / 0.8), 80))
    put_mask(f, text_mask("it collapses.", "grot", 78, 500, -0.02),
             120, 540, AMBER, fade(u, 2.6, 3.4, 8.0, 9.0), glow=0.5,
             reveal=(clamp((u - 2.6) / 0.8), 80))
    # temperature readout
    b = fade(u, 9.4, 10.6, 16.0, 17.2)
    temp = kf(u, [(9.4, 300.0), (14.0, 15.0)])
    put_mask(f, text_mask(f"{temp:05.1f} mK", "mono", 96, 300, 0.02),
             120, 430, ICE, b, glow=0.5)
    put_mask(f, text_mask("colder than deep space", "mono", 20, 400, 0.10),
             124, 540, CYAN, b * 0.85, glow=0.18)
    c = fade(u, 17.6, 18.6, 24.0, 24.8)
    put_mask(f, text_mask("fragile", "grot", 120, 300, -0.03),
             120, 470, WHITE, c * 0.9, glow=0.45, reveal=(clamp((u - 17.6) / 0.8), 90))
    put_mask(f, text_mask("error correction eats most of the machine", "mono", 19, 400, 0.05),
             124, 610, AMBER, fade(u, 19.0, 20.0, 24.0, 24.8) * 0.85, glow=0.15)


# ============================================================ 09. APPLICATIONS
def sc_apps(u, t):
    # three 5.67 s vignettes
    seg = int(u // 5.67)
    v = u - seg * 5.67
    cam = Cam(W, H, eye=(0.55 * math.sin(u * 0.25), 0.22, 3.4), target=(0, 0, 0), fov=42,
              roll=rad(1.2) * math.sin(u * 0.5))
    S = Scene(W, H, cam)
    rng = jitter(9 + seg)
    screen_glow(S, W * 0.5, H * 0.5, W * 0.32, CYAN * 0.45, 0.10)
    dot_ground(cam, S, t, n=4200, y=-1.05, extent=8, col=TEAL * 0.5, energy=0.16, blur=1.8)

    if seg == 0:
        # ---- molecule: two atom clusters, three bonds
        A = np.array([-0.55, 0.0, 0.0], F32)
        B = np.array([0.55, 0.0, 0.0], F32)
        spin = u * 0.5
        Rm = rot_y(spin) @ rot_x(rad(12))
        PA = sphere_shell(52000, 0.40, 0.02, rng) + A
        PB = sphere_shell(40000, 0.33, 0.02, rng) + B
        PA = PA @ Rm.T; PB = PB @ Rm.T
        cA = shade(PA, norm(PA - A @ Rm.T), CYAN * 0.6, [((1, 2, 2), ICE, 1.2)], cam.eye, amb=0.1, rim=1.1)
        cB = shade(PB, norm(PB - B @ Rm.T), AMBER * 0.55, [((1, 2, 2), ICE, 1.2)], cam.eye, amb=0.1, rim=1.1)
        S.dots(PA, cA, 0.55, blur=0.95); S.dots(PB, cB, 0.55, blur=0.95)
        for k in range(3):
            ph = k * TAU / 3 + 0.4
            off = np.array([0, 0.16 * math.sin(ph), 0.16 * math.cos(ph)], F32)
            bond = tube(np.stack([A @ Rm.T + off, B @ Rm.T + off]), 14000, 0.012, rng)
            S.dots(bond, TEAL, 0.45, blur=1.1)
    elif seg == 1:
        # ---- cryptography: a lattice cube that a key pierces
        n = 110000
        P = (rng.random((n, 3)) - 0.5) * 2.0
        # snap to edges of a cube lattice
        q = np.round(P * 2.0) / 2.0
        pick = rng.random(n) < 0.55
        P[pick] = q[pick] * 0.85
        Rm = rot_y(u * 0.55) @ rot_x(rad(18) + 0.15 * math.sin(u * 0.7))
        P = P @ Rm.T
        d = np.abs(P).max(1)
        S.dots(P, mix(TEAL, CYAN, clamp(d)), 0.42, blur=1.0)
        # key: a bright rod through the middle
        key = tube(np.stack([np.array([-1.3, 0, 0], F32), np.array([1.35, 0, 0], F32)]), 24000, 0.014, rng)
        key = key @ Rm.T
        S.dots(key, AMBER, 0.85, blur=1.0)
        # shatter near the end of the beat
        sh = sstep(4.0, 5.4, u - 5.67)
        if sh > 0:
            P2 = P * (1 + 0.35 * sh)
            S.dots(P2, AMBER, 0.35 * sh, blur=1.4)
    else:
        # ---- optimisation: a cost landscape with a descending point
        nx, nz = 120, 90
        P = grid3(nx, nz, 5.0, 4.0, 0.0)
        X, Z = P[:, 0], P[:, 2]
        h = (0.55 * np.exp(-((X - 1.1) ** 2 + (Z + 0.4) ** 2) * 1.8)
             + 0.8 * np.exp(-((X + 0.9) ** 2 + (Z - 0.7) ** 2) * 1.3)
             + 0.22 * np.sin(X * 2.4) * np.cos(Z * 2.0))
        P[:, 1] = h - 0.55
        col = mix(TEAL, AMBER, clamp(h / 1.2))
        S.dots(P, col, 0.30 + 0.35 * clamp(h), blur=1.1)
        # the descending probe
        q = ease_io(clamp((v - 0.4) / 3.4))
        px = lerp(-1.6, 1.1, q)
        pz = lerp(1.2, -0.4, q)
        ph = (0.55 * math.exp(-((px - 1.1) ** 2 + (pz + 0.4) ** 2) * 1.8)
              + 0.8 * math.exp(-((px + 0.9) ** 2 + (pz - 0.7) ** 2) * 1.3)
              + 0.22 * math.sin(px * 2.4) * math.cos(pz * 2.0)) - 0.55
        probe = sphere_volume(4000, 0.05, rng) + np.array([px, ph + 0.35 + 0.5 * (1 - q), pz], F32)
        S.dots(probe, WHITE, 1.2, blur=2.0)
        trail = np.stack([px + rng.normal(0, 0.02, 1200), ph + 0.35 + rng.normal(0, 0.15, 1200),
                          pz + rng.normal(0, 0.02, 1200)], 1)
        S.dots(trail, CYAN, 0.5, blur=2.2)
    dust_layer(S, t, 190, 2.9, energy=0.22)
    return S


def ov_apps(f, u, t, dur):
    seg = int(u // 5.67)
    v = u - seg * 5.67
    a = fade(v, 0.15, 0.7, 4.6, 5.3)
    label(f, "09 / APPLICATIONS", 120, 96, a * 0.85, 18)
    titles = [("molecules", "new drugs, designed atom by atom"),
              ("cryptography", "break it — then rebuild it"),
              ("optimisation", "find the lowest point in a vast landscape")]
    big, sub = titles[min(seg, 2)]
    put_mask(f, text_mask(big, "grot", 104, 300, -0.025),
             120, 430, WHITE, a, glow=0.45, reveal=(clamp((v - 0.15) / 0.7), 90))
    put_mask(f, text_mask(sub, "mono", 20, 400, 0.06),
             124, 560, CYAN * 0.9, fade(v, 0.8, 1.6, 4.6, 5.3) * 0.85, glow=0.18)
    idx = f"0{seg + 1}"
    put_mask(f, text_mask(idx, "mono", 20, 300, 0.10),
             1720, 96, ICE * 0.6, a * 0.7, glow=0.1)


# ================================================================ 10. SHADOW
def sc_shadow(u, t):
    cam = Cam(W, H, eye=(kf(u, [(0, -1.7), (18, 1.9)]), kf(u, [(0, 0.55), (18, 0.35)]),
                          kf(u, [(0, 3.0), (18, 3.6)])),
              target=(0.0, kf(u, [(0, 0.55), (18, 0.30)]), 0.0), fov=44,
              roll=rad(1.5) * math.sin(u * 0.3))
    S = Scene(W, H, cam)
    rng = jitter(10)
    screen_glow(S, W * 0.5, H * 0.34, W * 0.36, CYAN * 0.35, 0.12)

    # the quantum world above: a luminous structure
    n = 150000
    P = sphere_shell(n, 0.95, 0.02, rng)
    P[:, 1] *= 0.55
    P = P @ rot_y(u * 0.3).T
    P[:, 1] += 1.25
    c = shade(P, norm(P - np.array([0, 1.25, 0], F32)), CYAN * 0.55 + WHITE * 0.1,
              [((1.6, 3.2, 2.2), ICE, 1.2)], cam.eye, amb=0.08, rim=1.2, spec=0.5)
    S.dots(P, c, 0.55, blur=0.95)
    # rings above
    for k in range(2):
        R = ring3d(30000, 1.3 + k * 0.35, 0.01, tilt=rad(74 - k * 10), yaw=u * 0.25 * (1 + k), rng=rng)
        R[:, 1] += 1.2
        S.dots(R, TEAL, 0.35, blur=1.1)

    # the classical world below: its reflection, dimmer and desaturated
    Rp = P.copy(); Rp[:, 1] = -Rp[:, 1] + 0.02
    S.dots(Rp, mix(TEAL, INDIGO * 0.6, 0.5), 0.16, blur=2.6)
    for k in range(2):
        R = ring3d(22000, 1.3 + k * 0.35, 0.01, tilt=rad(74 - k * 10), yaw=u * 0.25 * (1 + k), rng=rng)
        R[:, 1] = -R[:, 1] + 0.02 + 2.4
        S.dots(R, TEAL * 0.6, 0.10, blur=3.0)

    # dotted floor = the mirror
    floor = grid3(90, 70, 9, 8, 0.0)
    d = np.abs(floor[:, 2])
    S.dots(floor, TEAL * 0.5, 0.10 * np.clip(1 - d / 8, 0.05, 1), blur=1.8)
    dust_layer(S, t, 220, 2.8, energy=0.24)
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


# ================================================================= 11. OUTRO
_PT_CACHE = {}

def title_points():
    if "P" not in _PT_CACHE:
        rng = np.random.default_rng(31)
        P = text_points("QUANTUM", "grot", 340, 500, tracking=0.01, n=150000, rng=rng)
        # scale to a comfortable world size
        P = P * 3.05
        P[:, 1] += 0.05
        P[:, 2] *= 2.0
        _PT_CACHE["P"] = P.astype(F32)
    return _PT_CACHE["P"]


def sc_outro(u, t):
    cam = Cam(W, H, eye=(0.0, 0.0, kf(u, [(0, 5.4), (7, 4.55)])),
              target=(0.0, 0.0, 0.0), fov=38, roll=0.0)
    S = Scene(W, H, cam)
    rng = jitter(11)
    screen_glow(S, W * 0.5, H * 0.5, W * 0.5, CYAN * 0.25, 0.10)
    Pt = title_points()
    n = len(Pt)
    # start as a sphere, assemble into the word
    start = sphere_shell(n, 2.4, 0.25, rng)
    q = ease_io(clamp((u - 0.2) / 3.4))
    # per-particle stagger for a hand-assembled feel
    stag = rng.random(n) * 0.35
    qq = clamp((q - stag) / (1 - stag))[:, None]
    P = start * (1 - qq) + Pt * qq
    # gentle drift once assembled
    P = P + np.stack([np.sin(u * 0.7 + Pt[:, 0] * 2) * 0.012,
                      np.sin(u * 0.9 + Pt[:, 1] * 2) * 0.012,
                      np.zeros(n, F32)], 1) * q
    # breathing glow
    pulse = 0.75 + 0.25 * math.sin(u * 1.6)
    col = mix(CYAN, WHITE, 0.35 + 0.35 * np.sin(Pt[:, 0] * 1.5 + u * 0.6))
    S.dots(P, col, (0.55 + 0.5 * q) * pulse, blur=0.95)
    # halo behind the word
    hz = P * 1.0
    S.dots(hz + np.array([0, 0, -0.4], F32), TEAL, 0.10 * q, blur=4.0)
    dust_layer(S, t, 160, 3.0, energy=0.18)
    return S


def ov_outro(f, u, t, dur):
    a = fade(u, 3.6, 4.6, 10.2, 11.2)
    put_mask(f, text_mask("a new language.", "serif_i", 54, None),
             120, 800, ICE, a * 0.95, glow=0.35)
    label(f, "11 / QUANTUM", 120, 96, fade(u, 4.0, 5.0, 10.2, 11.2) * 0.8, 18)
    # end mark
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


def fade(u, t0, t1, t2, t3):
    return sstep(t0, t1, u) * (1.0 - sstep(t2, t3, u))


def build(t):
    idx, u, u0, dur = locate(t)
    return SCENES[idx](u, t)


def render_scene(t, nsub=NSUB):
    acc = None
    for tt in shutter_samples(t, FPS, nsub):
        S = build(tt)
        img = S.compose()
        acc = img if acc is None else acc + img
    return acc / max(len(shutter_samples(t, FPS, nsub)), 1)


def render_frame(t):
    img = render_scene(t, NSUB)
    frame = finish(img, OUT_W, OUT_H, exposure=1.05, grain=0.014, vig=0.5, ca=0.0014, t=t)
    f = frame.astype(F32) / 255.0
    idx, u, u0, dur = locate(t)
    OVERLAYS[idx](f, u, t, dur)
    # cut flash
    if idx in FLASH:
        k = math.exp(-u * 26.0)
        f = f + FLASH[idx] * k
    # section-boundary dip for the finale
    f = np.clip(f, 0, 1)
    return (f * 255.0 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    import time, sys
    t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    st = time.time()
    fr = render_frame(t0)
    print("render", round(time.time() - st, 2), "s", fr.shape)
