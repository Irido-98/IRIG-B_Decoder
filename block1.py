import pandas as pd
import numpy as np

# Path to your captured waveform. Change this to wherever you saved it.
CSV_PATH = "Ch1FullFrame.csv"

# Read the file. The waveform sits in columns 4 and 5 (1-indexed),
# which are index 3 and 4 to pandas. There is no header row, so we
# tell pandas not to treat the first line as column names.
raw = pd.read_csv(CSV_PATH, header=None, usecols=[3, 4])

# Pull the two columns out into numpy arrays.
time = raw[3].to_numpy()
voltage = raw[4].to_numpy()

# Sanity check: print what we got so we can confirm it is sensible
# before trusting any later code with it.
print(f"Loaded {len(voltage)} samples")
print(f"Time range: {time[0]:.4f} s to {time[-1]:.4f} s")
print(f"Voltage range: {voltage.min():.3f} V to {voltage.max():.3f} V")

# --- Block 2: thresholding ---
# Convert the analogue voltage samples into a clean high/low logic
# signal. Anything above the threshold is "high" (True), anything
# below is "low" (False).
THRESHOLD = 2.5  # volts; sits in the middle of the ~0 V to ~5 V swing
logic = voltage > THRESHOLD

# Sanity check: roughly how much of the capture is high vs low?
print(f"Samples high: {logic.sum()} of {len(logic)} ({100*logic.sum()/len(logic):.1f}%)")

# Find edges by looking at where the logic signal changes.
# Convert True/False to 1/0 first so we can do arithmetic on it.
logic_int = logic.astype(int)

# np.diff gives the difference between each sample and the one before.
# A rising edge (0 -> 1) shows up as +1; a falling edge (1 -> 0) as -1;
# no change shows up as 0.
changes = np.diff(logic_int)

# The indices where a rising edge happened (difference of +1)
rising_edges = np.where(changes == 1)[0] + 1

# The indices where a falling edge happened (difference of -1)
falling_edges = np.where(changes == -1)[0] + 1

print(f"Found {len(rising_edges)} rising edges and {len(falling_edges)} falling edges")

# The capture started mid-pulse, so the first falling edge has no
# matching rising edge. Drop it so the two lists pair up index-for-index.
if len(falling_edges) > len(rising_edges):
    falling_edges = falling_edges[1:]

# If the capture instead ended mid-pulse (a final rising edge with no
# falling edge after it), drop that unpaired rising edge.
if len(rising_edges) > len(falling_edges):
    rising_edges = rising_edges[:-1]

print(f"After trimming: {len(rising_edges)} rising, {len(falling_edges)} falling")

# Width of each pulse = time at its falling edge minus time at its rising edge.
# rising_edges and falling_edges are now aligned, so element i of each
# refers to the same pulse.
pulse_widths = time[falling_edges] - time[rising_edges]

# Convert to milliseconds for readability (IRIG-B widths are 2/5/8 ms).
pulse_widths_ms = pulse_widths * 1000

print(f"Measured {len(pulse_widths_ms)} pulse widths")
print(f"Min width: {pulse_widths_ms.min():.3f} ms")
print(f"Max width: {pulse_widths_ms.max():.3f} ms")
print(f"First 10 widths (ms): {pulse_widths_ms[:10]}")

# Classify each pulse width into a symbol.
#   < 3.5 ms  -> binary 0   (nominal 2 ms)
#   3.5-6.5ms -> binary 1   (nominal 5 ms)
#   > 6.5 ms  -> marker     (nominal 8 ms)
symbols = []
for w in pulse_widths_ms:
    if w < 3.5:
        symbols.append('0')
    elif w < 6.5:
        symbols.append('1')
    else:
        symbols.append('M')

print(f"Classified {len(symbols)} symbols")
print(f"Symbol string: {''.join(symbols)}")

# ============================================================
# Block 3: frame synchronisation and time decode
# ============================================================

# --- SWITCH: choose how many frames to decode ---
#   "single" -> decode only the first complete frame
#   "all"    -> decode every complete frame in the capture
DECODE_MODE = "decode"   # change to "all" to decode every frame


def find_frame_starts(symbols):
    """Return the index of each reference marker (the 2nd M of an MM pair).
    The frame's data bits begin at the position AFTER this index."""
    starts = []
    for i in range(len(symbols) - 1):
        if symbols[i] == 'M' and symbols[i + 1] == 'M':
            starts.append(i + 1)   # i+1 is Pr, the reference marker
    return starts


def bcd_field(bits, weights):
    """Decode one little-endian BCD field.
    bits:    list of '0'/'1' characters for this field
    weights: the value of each bit position, e.g. [1,2,4,8]
    Returns the decimal value of the field."""
    value = 0
    for bit, weight in zip(bits, weights):
        if bit == '1':
            value += weight
    return value


def decode_frame(symbols, start):
    """Decode the time-of-day from one frame.
    'start' is the index of the reference marker Pr (position 0).
    Data bits are at start+1, start+2, ... relative to Pr."""
    # Pull out 100 bit-slots following the reference marker. We index
    # relative to 'start' so position p in the frame is symbols[start + p].
    def bit(p):
        return symbols[start + p]

    # Seconds: units in bits 1-4, tens in bits 6-8 (little-endian)
    sec_units = bcd_field([bit(1), bit(2), bit(3), bit(4)], [1, 2, 4, 8])
    sec_tens  = bcd_field([bit(6), bit(7), bit(8)],          [10, 20, 40])
    seconds = sec_tens + sec_units

    # Minutes: units in bits 10-13, tens in bits 15-17
    min_units = bcd_field([bit(10), bit(11), bit(12), bit(13)], [1, 2, 4, 8])
    min_tens  = bcd_field([bit(15), bit(16), bit(17)],           [10, 20, 40])
    minutes = min_tens + min_units

    # Hours: units in bits 20-23, tens in bits 25-26
    hour_units = bcd_field([bit(20), bit(21), bit(22), bit(23)], [1, 2, 4, 8])
    hour_tens  = bcd_field([bit(25), bit(26)],                    [10, 20])
    hours = hour_tens + hour_units

    return hours, minutes, seconds


# --- Run it according to the switch ---
frame_starts = find_frame_starts(symbols)
print(f"Found {len(frame_starts)} frame start(s) at symbol indices: {frame_starts}")

if DECODE_MODE == "single":
    # Decode just the first complete frame. A frame needs 100 bits after
    # Pr, so only attempt it if enough symbols follow the marker.
    decoded = False
    for start in frame_starts:
        if start + 26 < len(symbols):   # need at least up to bit 26 (hours)
            h, m, s = decode_frame(symbols, start)
            print(f"Decoded time: {h:02d}:{m:02d}:{s:02d}")
            decoded = True
            break
    if not decoded:
        print("No complete frame found in capture.")

elif DECODE_MODE == "all":
    for start in frame_starts:
        if start + 26 < len(symbols):
            h, m, s = decode_frame(symbols, start)
            print(f"Frame at index {start}: {h:02d}:{m:02d}:{s:02d}")