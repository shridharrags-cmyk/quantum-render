# Narration

Voice: `en-US-AndrewNeural` (edge-tts), synthesized per section at a uniform rate
(+0%, except s04 +3% and s07 +2%), then placed at the film's section boundaries
with silence padding - never time-stretched.

Rebuild:

```bash
for n in s00 s01 s02 s03 s04 s05 s06 s07 s08 s09 s10; do
  edge-tts -f narration/$n.txt -v en-US-AndrewNeural --rate=+0% \
    --write-media $n.mp3 --write-subtitles $n.vtt --no-progress
done
# trim leading/trailing silence, normalise each section to the same RMS,
# place at SEC[i], master with highpass + presence EQ + compressor + loudnorm
```

The section texts are in `script.md`. `audio/narr_final.wav` is the mastered
192.000 s mix (voice + ambient bed) that the render workflow muxes.
