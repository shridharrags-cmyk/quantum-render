"""Render one shard of the film and encode it as a video segment."""
import os
import sys
import time
import subprocess
import multiprocessing as mp

os.environ.setdefault("QFX_NSUB", "2")

FPS = 24
TOTAL = 4608


def work(args):
    i, first = args
    import film
    from PIL import Image
    fr = film.render_frame(i / FPS)
    Image.fromarray(fr).save(f"frames/f{i - first:05d}.png", compress_level=3)
    return i


def main():
    shard = int(sys.argv[1])
    nshards = int(sys.argv[2])
    per = TOTAL // nshards
    first = shard * per
    last = TOTAL if shard == nshards - 1 else (shard + 1) * per
    os.makedirs("frames", exist_ok=True)
    t0 = time.time()
    nproc = min(4, os.cpu_count() or 2)
    print(f"shard {shard}/{nshards}: frames {first}..{last-1} on {nproc} procs", flush=True)
    done = 0
    with mp.Pool(nproc) as p:
        for i in p.imap_unordered(work, [(i, first) for i in range(first, last)], chunksize=2):
            done += 1
            if done % 96 == 0:
                el = time.time() - t0
                print(f"  {done}/{last-first} frames  {el:6.1f}s  "
                      f"({done/el:.2f} fps)", flush=True)
    n = len(os.listdir("frames"))
    print(f"rendered {n} frames in {time.time()-t0:.0f}s", flush=True)
    if n != last - first:
        print(f"::error frame count {n} != {last-first}")
        sys.exit(1)
    seg = f"seg_{shard:02d}.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
           "-i", "frames/f%05d.png", "-an",
           "-c:v", "libx264", "-preset", "fast", "-crf", "16",
           "-pix_fmt", "yuv420p", "-g", "48", "-movflags", "+faststart", seg]
    subprocess.run(cmd, check=True)
    print("segment:", seg, os.path.getsize(seg) // 1024 // 1024, "MB", flush=True)


if __name__ == "__main__":
    main()
