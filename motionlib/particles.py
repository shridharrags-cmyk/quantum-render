"""motionlib.particles — flow fields, vectorized particle systems, kaleidoscopes."""
import numpy as np

# ---------------- value noise ----------------
def _hash(ix, iy, seed=0):
    h = (ix * 374761393 + iy * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFFFF) / 0xFFFFFFFF

def value_noise(x, y, seed=0):
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    n00, n10 = _hash(ix, iy, seed), _hash(ix + 1, iy, seed)
    n01, n11 = _hash(ix, iy + 1, seed), _hash(ix + 1, iy + 1, seed)
    nx0 = n00 * (1 - ux) + n10 * ux
    nx1 = n01 * (1 - ux) + n11 * ux
    return nx0 * (1 - uy) + nx1 * uy

def curl(x, y, eps=1e-3, seed=0, t=0.0):
    """Curl of a scalar noise field -> divergence-free 2D flow."""
    n1 = value_noise(x, y + eps, seed) - value_noise(x, y - eps, seed)
    n2 = value_noise(x + eps, y, seed) - value_noise(x - eps, y, seed)
    return n2 / (2 * eps), -n1 / (2 * eps)

# ---------------- particle system ----------------
class FlowParticles:
    """N particles advected through a curl-noise flow field with trails."""
    def __init__(self, n, w, h, seed=1, speed=90.0, life=(0.8, 2.2)):
        self.n, self.w, self.h = n, w, h
        self.rng = np.random.default_rng(seed)
        self.speed = speed
        self.life = life
        self.reset_all()
        self.age = self.rng.uniform(0, 1, n) * self.life_lo

    def reset_all(self):
        r = self.rng
        self.x = r.uniform(0, self.w, self.n)
        self.y = r.uniform(0, self.h, self.n)
        self.life_lo = np.full(self.n, self.life[0])
        self.life_span = r.uniform(self.life[0], self.life[1], self.n)

    def step(self, dt, t, fade=0.90, ink=1.0):
        rng_now = self.rng
        vx, vy = curl(self.x / 220.0, self.y / 220.0 + t * 0.05, seed=int(t * 7) % 97)
        self.x = (self.x + vx * self.speed * dt) % self.w
        self.y = (self.y + vy * self.speed * dt) % self.h
        self.age += dt
        dead = self.age > self.life_span
        if dead.any():
            self.x[dead] = rng_now.uniform(0, self.w, dead.sum())
            self.y[dead] = rng_now.uniform(0, self.h, dead.sum())
            self.age[dead] = 0.0
            self.life_span[dead] = rng_now.uniform(self.life[0], self.life[1], dead.sum())
        return self.x, self.y, (1 - self.age / self.life_span) * ink

# ---------------- kaleidoscope ----------------
def kaleido(src, segments=12, center=None, rot=0.0):
    """Polar mirror fold: wedge-copy + flip around `segments` radial axes."""
    h, w = src.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - center[0], yy - center[1]
    ang = (np.arctan2(dy, dx) + rot) % (2 * np.pi)
    wedge = (ang / (2 * np.pi)) * segments
    fold = np.abs(((wedge + 0.5) % 1.0) - 0.5) * 2  # triangle wave in wedge space
    src_ang = fold * (2 * np.pi / segments) - rot
    # sample source by remapped angle, same radius
    r = np.sqrt(dx * dx + dy * dy)
    sx = (center[0] + r * np.cos(src_ang)).astype(np.int32)
    sy = (center[1] + r * np.sin(src_ang)).astype(np.int32)
    sx = np.clip(sx, 0, w - 1); sy = np.clip(sy, 0, h - 1)
    out = src[sy, sx]
    return out

# ---------------- shapes ----------------
def poly_mask(shape, cx, cy, radius, sides, rot=0.0, width=3, aa=1.5):
    """Stroke polygon -> float mask."""
    from PIL import Image, ImageDraw
    h, w = shape
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    pts = [(cx + radius * np.cos(rot + i * 2 * np.pi / sides),
            cy + radius * np.sin(rot + i * 2 * np.pi / sides)) for i in range(sides)]
    d.line(pts + [pts[0]], fill=255, width=width, joint="curve")
    return np.asarray(img).astype(np.float32) / 255.0

def disk_mask(shape, cx, cy, radius, soft=2.0):
    from PIL import Image, ImageDraw
    from .core import box_blur
    h, w = shape
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    r = max(1.0, radius)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    m = np.asarray(img).astype(np.float32) / 255.0
    if soft > 0:
        m = box_blur(m, max(1, int(soft)))
    return m

def ring_mask(shape, cx, cy, radius, thickness=6.0, soft=2.0):
    from .core import box_blur
    return rings_batch(shape, [(cx, cy, radius, thickness)], soft)

def rings_batch(shape, specs, soft=2):
    """specs: [(cx, cy, r, thickness), ...] -> one mask, one blur."""
    from PIL import Image, ImageDraw
    from .core import box_blur
    h, w = shape
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for cx, cy, r, th in specs:
        r, th = max(1.0, r), max(1.0, th)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=int(th))
    m = np.asarray(img).astype(np.float32) / 255.0
    if soft > 0:
        m = box_blur(m, max(1, int(soft)))
    return m

def disks_batch(shape, specs):
    """specs: [(cx, cy, r), ...] -> one mask (no blur)."""
    from PIL import Image, ImageDraw
    h, w = shape
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for cx, cy, r in specs:
        r = max(1.0, r)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return np.asarray(img).astype(np.float32) / 255.0