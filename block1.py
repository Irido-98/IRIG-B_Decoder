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

