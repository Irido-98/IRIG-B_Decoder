import pandas as pd

CSV_PATH = "Ch3FullFrame000.csv"

raw = pd.read_csv(CSV_PATH, header=None, usecols=[3, 4])
time = raw[3].tolist()
voltage = raw[4].tolist()

# check the voltage range to determine threshold
print(f"Voltage range: {min(voltage):.3f} V to {max(voltage):.3f} V")

THRESHOLD = 2.5

# turn each voltage sample into True (high) or False (low)
logic = []
for v in voltage:
    logic.append(v > THRESHOLD)

# walk through the signal and record where it switches state
rising_edges = []
falling_edges = []
for i in range(1, len(logic)):
    if logic[i] and not logic[i - 1]:      # low last sample, high now = rising
        rising_edges.append(i)
    elif not logic[i] and logic[i - 1]:    # high last sample, low now = falling
        falling_edges.append(i)

print(f"Found {len(rising_edges)} rising edges and {len(falling_edges)} falling edges")

# incase capture started early, trim any misaligned edges
if len(falling_edges) > len(rising_edges):
    falling_edges = falling_edges[1:]
if len(rising_edges) > len(falling_edges):
    rising_edges = rising_edges[:-1]

# pulse width in ms
pulse_widths_ms = []
for i in range(len(rising_edges)):
    width = (time[falling_edges[i]] - time[rising_edges[i]]) * 1000
    pulse_widths_ms.append(width)

print(f"Max width: {max(pulse_widths_ms):.3f} ms")  # should be ~8ms or thresholds are off

symbols = []
for w in pulse_widths_ms:
    if w < 3.5:
        symbols.append('0')
    elif w < 6.5:
        symbols.append('1')
    else:
        symbols.append('M')

print(f"Symbol string: {''.join(symbols)}")