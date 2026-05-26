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