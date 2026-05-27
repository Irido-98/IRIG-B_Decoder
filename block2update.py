import pandas as pd
import numpy as np

CSV_PATH = "Ch1FullFrame.csv"

# scope dumps the waveform in cols 4 and 5, no header
raw = pd.read_csv(CSV_PATH, header=None, usecols=[3, 4])
time = raw[3].to_numpy()
voltage = raw[4].to_numpy()

print(f"Loaded {len(voltage)} samples")
print(f"Time range: {time[0]:.4f} s to {time[-1]:.4f} s")
print(f"Voltage range: {voltage.min():.3f} V to {voltage.max():.3f} V")

# threshold halfway up the 0-5V swing
THRESHOLD = 2.5
logic = voltage > THRESHOLD
print(f"Samples high: {logic.sum()} of {len(logic)} ({100*logic.sum()/len(logic):.1f}%)")

# +1 because diff is offset by one sample relative to the original array
changes = np.diff(logic.astype(int))
rising_edges = np.where(changes == 1)[0] + 1
falling_edges = np.where(changes == -1)[0] + 1
print(f"Found {len(rising_edges)} rising edges and {len(falling_edges)} falling edges")

# capture started/ended mid-pulse so the edge lists won't line up, trim the orphans
if len(falling_edges) > len(rising_edges):
    falling_edges = falling_edges[1:]
if len(rising_edges) > len(falling_edges):
    rising_edges = rising_edges[:-1]
print(f"After trimming: {len(rising_edges)} rising, {len(falling_edges)} falling")

pulse_widths_ms = (time[falling_edges] - time[rising_edges]) * 1000
print(f"Measured {len(pulse_widths_ms)} pulse widths")
print(f"Min width: {pulse_widths_ms.min():.3f} ms")
print(f"Max width: {pulse_widths_ms.max():.3f} ms")
print(f"First 10 widths (ms): {pulse_widths_ms[:10]}")

# IRIG-B: 2ms=0, 5ms=1, 8ms=marker. split at the gaps between those
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