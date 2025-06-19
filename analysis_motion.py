import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

df = pd.read_csv('/Volumes/JData5/JPeople/Florian/treadmill/motion_test/10s_10r_10s_10l_20250619_150939.csv')
df['timestamp'] = df['timestamp']/1000
df['timestamp'] = df['timestamp'] - df['timestamp'].min()

df2 = df
x = df2['posx'].values* 0.45
y = df2['posy'].values* 0.45
t = df2['frame_counter'].values-min(df2['frame_counter'].values)
ts = df2['timestamp'].values-min(df2['timestamp'].values)
err = df2['delta_rot_err']
seq = df2['seq_counter'].values


third = ts
third_str = 'Time (s)'

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(third.min(), third.max()))
lc.set_array(third)
lc.set_linewidth(2)

fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
plt.colorbar(lc, label=third_str)
plt.xlabel('posx (cm)')
plt.ylabel('posy (cm)')
plt.show()



