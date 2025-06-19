import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

df = pd.read_csv('fictrac_log_20250619_110412.csv')

# Create or reset the 'seq_no' column
df['seq_no'] = np.nan

# Identify where a new sequence starts
sequence_starts = df.loc[df['seq_counter'] == 0].index.to_list()

# Assign sequence numbers
for i, ss in enumerate(sequence_starts):
    if i < len(sequence_starts) - 1:
        # Fill seq_no from this start to just before the next start
        df.loc[ss:sequence_starts[i + 1] - 1, 'seq_no'] = i
    else:
        # Fill from last start to the end of the DataFrame
        df.loc[ss:, 'seq_no'] = i

# Optional: convert seq_no to integer if needed
df['seq_no'] = df['seq_no'].astype(int)

# Access the last row of each sequence
for seq in df['seq_no'].unique():
    if seq == 0:
        continue

    last_row = df.loc[df['seq_no'] == seq-1].iloc[-1]
    temp_x = last_row['posx']
    temp_y = last_row['posy']

    df.loc[df['seq_no'] == seq, 'posx'] = df.loc[df['seq_no'] == seq, 'posx'] + temp_x
    df.loc[df['seq_no'] == seq, 'posy'] = df.loc[df['seq_no'] == seq, 'posy'] + temp_y



df2 = df
x = df2['posx'].values
y = df2['posy'].values
t = df2['frame_counter'].values-min(df2['frame_counter'].values)
err = df2['delta_rot_err']
seq = df2['seq_counter'].values


third = t
third_str = 'Time (frame counter)'

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(third.min(), third.max()))
lc.set_array(third)
lc.set_linewidth(2)

fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
plt.colorbar(lc, label=third_str)
plt.xlabel('posx')
plt.ylabel('posy')
plt.show()