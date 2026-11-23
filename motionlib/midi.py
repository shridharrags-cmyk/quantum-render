"""audio/midi.py — minimal format-0 MIDI writer (no dependencies).

Enough for what FluidSynth needs: tempo, program changes, notes with velocity.
"""
import struct

def vlq(n):
    b = [n & 0x7F]
    n >>= 7
    while n:
        b.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(b))

def write_midi(path, bpm, events, division=480):
    """events: list of (tick, message_bytes). Sorted internally."""
    evs = sorted(events, key=lambda e: e[0])
    track = bytearray()
    track += b"\x00\xff\x51\x03" + struct.pack(">I", int(60_000_000 / bpm))[1:]
    last = 0
    for tick, msg in evs:
        track += vlq(tick - last) + msg
        last = tick
    track += b"\x00\xff\x2f\x00"
    data = b"MThd" + struct.pack(">IHHH", 6, 0, 1, division) + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    open(path, "wb").write(data)

def sec_to_tick(t, bpm, division=480):
    return int(round(t * bpm / 60 * division))

def note_on(ch, note, vel):
    return bytes([0x90 | ch, note, max(1, min(127, int(vel)))])

def note_off(ch, note):
    return bytes([0x80 | ch, note, 0])

def program(ch, prog):
    return bytes([0xC0 | ch, prog])