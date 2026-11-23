"""motionlib.text — kinetic typography on numpy masks."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .core import FONT_DIR, to_img

_cache = {}

def glyph_mask(ch, size, font="bold", tracking_gap=0):
    """Render one character to a float32 'L' mask, cached."""
    key = (ch, size, font)
    if key in _cache:
        return _cache[key]
    fpath = {
        "bold": f"{FONT_DIR}/DejaVuSans-Bold.ttf",
        "sans": f"{FONT_DIR}/DejaVuSans.ttf",
        "cond": f"{FONT_DIR}/DejaVuSansCondensed-Bold.ttf",
        "mono": f"{FONT_DIR}/DejaVuSansMono-Bold.ttf",
        "serif": f"{FONT_DIR}/DejaVuSerif-Bold.ttf",
    }[font]
    f = ImageFont.truetype(fpath, size)
    if ch == " ":
        # real advance width so words don't collapse together
        m = np.zeros((max(1, size // 3), max(1, int(size * 0.5))), np.float32)
        _cache[key] = m
        return m
    img = Image.new("L", (size * 3, size * 3), 0)
    d = ImageDraw.Draw(img)
    d.text((size, size), ch, font=f, fill=255)
    bbox = img.getbbox()
    if bbox is None:
        m = np.zeros((1, 1), np.float32)
    else:
        m = np.asarray(img.crop(bbox)).astype(np.float32) / 255.0
    _cache[key] = m
    return m

def text_line(word, size, font="bold", spacing=-0.02):
    """Render a whole word/line -> (mask, per-char x-offsets, widths)."""
    chars = list(word)
    masks = [glyph_mask(c, size, font) for c in chars]
    widths = [m.shape[1] for m in masks]
    gap = int(size * spacing)
    xs, x = [], 0
    for w in widths:
        xs.append(x)
        x += w + gap
    total_w = max(1, x - gap)
    H = max(m.shape[0] for m in masks)
    line = np.zeros((H, total_w), np.float32)
    for m, x0 in zip(masks, xs):
        line[:m.shape[0], x0:x0 + m.shape[1]] = np.maximum(
            line[:m.shape[0], x0:x0 + m.shape[1]], m)
    return line, xs, widths, masks

def place(mask, hw, hh, cx, cy, rot=0.0, scale=1.0, alpha=1.0):
    """Return a full-frame placement of a char mask via PIL affine."""
    H, W = hh, hw
    m = mask
    if scale != 1.0:
        im = Image.fromarray((m * 255).astype(np.uint8), "L")
        im = im.resize((max(1, int(m.shape[1] * scale)), max(1, int(m.shape[0] * scale))), Image.LANCZOS)
        m = np.asarray(im).astype(np.float32) / 255.0
    if rot != 0.0:
        im = Image.fromarray((m * 255).astype(np.uint8), "L")
        im = im.rotate(rot, expand=True, resample=Image.BICUBIC)
        m = np.asarray(im).astype(np.float32) / 255.0
    h, w = m.shape
    canvas = np.zeros((H, W), np.float32)
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return canvas
    canvas[sy0:sy1, sx0:sx1] = m[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] * alpha
    return canvas

def glow(mask, radius=6, gain=1.6):
    from .core import box_blur
    return np.clip(box_blur(mask[..., None], radius)[..., 0] * gain, 0, 1)