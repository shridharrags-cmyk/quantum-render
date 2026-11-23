"""Render a set of preview stills (full resolution) for art review."""
import os
import sys
import time
import multiprocessing as mp

os.environ.setdefault("QFX_NSUB", "1")

TIMES = [3.0, 8.0, 18.0, 25.0, 31.0, 36.0, 44.0, 52.0, 58.0, 66.0, 78.0,
         88.0, 100.0, 112.0, 122.0, 132.0, 140.0, 150.0, 157.0, 166.0,
         176.0, 186.0]


def work(t):
    import film
    from PIL import Image
    st = time.time()
    fr = film.render_frame(t)
    os.makedirs("previews", exist_ok=True)
    Image.fromarray(fr).save(f"previews/t{t:06.1f}.png")
    return t, time.time() - st


if __name__ == "__main__":
    ts = [float(x) for x in sys.argv[1:]] or TIMES
    with mp.Pool(min(4, os.cpu_count() or 2)) as p:
        for t, dt in p.imap_unordered(work, ts):
            print(f"t={t:6.1f}  {dt:5.2f}s", flush=True)
    print("STILLS DONE", len(ts))
