"""Parallel renderer: N workers each render a frame range to PNG, then one encode."""
import sys, os, time, subprocess
sys.path.insert(0, ".")
sys.path.insert(0, "/tmp/opencode/quantum")
import multiprocessing as mp

FPS, TOTAL = 24, 4608
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")

def worker(args):
    a, b = args
    import render_quantum as R
    from motionlib.core import to_img
    done = 0
    for i in range(a, b):
        p = f"{OUT}/f{i:05d}.png"
        if os.path.exists(p):
            continue
        to_img(R.frame_at(i / FPS)).save(p)
        done += 1
    return done

def main():
    os.makedirs(OUT, exist_ok=True)
    n = min(8, mp.cpu_count())
    chunk = (TOTAL + n - 1) // n
    ranges = [(i * chunk, min((i + 1) * chunk, TOTAL)) for i in range(n)]
    t0 = time.time()
    with mp.Pool(n) as pool:
        for k, d in enumerate(pool.imap_unordered(worker, ranges)):
            out(f"chunk {k} done ({d} new frames) {time.time()-t0:.0f}s")
    out(f"ALL FRAMES DONE {time.time()-t0:.0f}s")

def out(s):
    print(s, flush=True)

if __name__ == "__main__":
    main()
