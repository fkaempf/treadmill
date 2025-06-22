import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/fkampf/Downloads/10s_10r_10s_10l_20250620_092909.csv')
df['timestamp'] = df['timestamp']/1000
df['timestamp'] = df['timestamp'] - df['timestamp'].min()
df
epochs = int((df['timestamp'].max()-df['timestamp'].max()%10)/10)

def get_stim_seq():
    seq = ['static', 'left', 'static', 'right']
    while True:
        yield from seq
a = get_stim_seq()

def zero_radian(s):
    s = s - s.iloc[0]
    s = s.apply(lambda x: x if x >= 0 else x + 2 * np.pi)
    return s

def write_delta(s):
    temp = []
    for i,item in s.items():
        
        if i == s.index.min():
            temp.append(0)
        else:
            temp.append(item - s[i-1])


#interpolate
def interpolate(arr,target_length=700):
    x_original = np.linspace(0, 1, len(arr))
    x_target = np.linspace(0, 1, target_length)
    f = interp1d(x_original, arr, kind='linear', fill_value='extrapolate')
    return f(x_target)

trialy_array = None
trialx_array = None  # <-- Add this line
trial_heading_array = None
array_stim = []
for i in range(epochs):
    temp_stim = next(a)
    df.loc[(df['timestamp'] >= i*10) & (df['timestamp'] < (i+1)*10), 'epoch'] = i
    df.loc[(df['epoch'] == i), 'stim'] = temp_stim

    df.loc[(df['epoch'] == i), 'trial_time'] = df.loc[(df['epoch'] == i), 'timestamp'] - df.loc[df['epoch'] == i, 'timestamp'].iloc[0]
    df.loc[(df['epoch'] == i), 'trial_heading'] = np.unwrap(zero_radian(df.loc[(df['epoch'] == i), 'heading']))
    df.loc[(df['epoch'] == i), 'trial_posy'] = df.loc[(df['epoch'] == i), 'posy'] - df.loc[df['epoch'] == i, 'posy'].iloc[0]
    df.loc[(df['epoch'] == i), 'trial_posx'] = df.loc[(df['epoch'] == i), 'posx'] - df.loc[df['epoch'] == i, 'posx'].iloc[0]  # <-- Add this line

    if i == epochs-1:
        df.loc[(df['timestamp'] >= i*10), 'epoch'] = i
        df.loc[(df['epoch'] == i), 'stim'] = temp_stim
        df.loc[(df['epoch'] == i), 'timestamp'] - df.loc[df['epoch'] == i, 'timestamp'].iloc[0]
        df.loc[(df['epoch'] == i), 'trial_posy'] = df.loc[(df['epoch'] == i), 'posy'] - df.loc[df['epoch'] == i, 'posy'].iloc[0]
        df.loc[(df['epoch'] == i), 'trial_posx'] = df.loc[(df['epoch'] == i), 'posx'] - df.loc[df['epoch'] == i, 'posx'].iloc[0]  # <-- Add this line

    if min(interpolate(df.loc[df['epoch'] == i, 'trial_posy'].values)) > -10:
        array_stim.append(temp_stim)
        if trialy_array is None:
            trialy_array = interpolate(df.loc[df['epoch'] == i, 'trial_posy'].values)
        else:
            trialy_array = np.vstack((trialy_array, interpolate(df.loc[df['epoch'] == i, 'trial_posy'].values)))

        # trialx_array setup (same logic as trialy_array)
        if trialx_array is None:
            trialx_array = interpolate(df.loc[df['epoch'] == i, 'trial_posx'].values)
        else:
            trialx_array = np.vstack((trialx_array, interpolate(df.loc[df['epoch'] == i, 'trial_posx'].values)))

    if min(interpolate(df.loc[df['epoch'] == i, 'trial_posy'].values)) > -10:
        if trial_heading_array is None:
            trial_heading_array = interpolate(df.loc[df['epoch'] == i, 'trial_heading'].values)
        else:
            trial_heading_array = np.vstack((trial_heading_array, interpolate(df.loc[df['epoch'] == i, 'trial_heading'].values)))
array_stim = np.array(array_stim)

plt.plot(np.mean(trialy_array[array_stim=='left',:],axis=0),label='left')
plt.plot(np.mean(trialy_array[array_stim=='static',:],axis=0), label='static')
plt.plot(np.mean(trialy_array[array_stim=='right',:],axis=0), label='right')
plt.legend()
plt.show()

plt.plot(trialx_array[array_stim=='left',:].T,
         trialy_array[array_stim=='left',:].T,label='left',c='red', alpha=0.3)
plt.plot(trialx_array[array_stim=='static',:].T,
         trialy_array[array_stim=='static',:].T, label='static',c='gray', alpha=0.1)
plt.plot(trialx_array[array_stim=='right',:].T,
         trialy_array[array_stim=='right',:].T, label='right',c='blue', alpha=0.3)

plt.show()

def plot_mean_sem(data, condition_mask, label, color=None):
    subset = data[condition_mask, :]
    mean = np.nanmean(subset, axis=0)
    sem = np.nanstd(subset, axis=0) / np.sqrt(np.sum(condition_mask))
    plt.plot(seconds, mean, label=label, color=color)
    plt.fill_between(seconds, mean - sem, mean + sem, alpha=0.3, color=color)


num_frames = 700
seconds = np.linspace(0, 10, num_frames) 

# Plot each condition with SEM
plot_mean_sem(trial_heading_array, array_stim == 'left', f'left n = {len(df.loc[df['stim']=='left','epoch'].unique())}', color='red')
plot_mean_sem(trial_heading_array, array_stim == 'static', f'static n = {len(df.loc[df['stim']=='static','epoch'].unique())}', color='gray')
plot_mean_sem(trial_heading_array, array_stim == 'right', f'right n = {len(df.loc[df['stim']=='right','epoch'].unique())}', color='blue')

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Heading")
plt.title("Heading over Time with SEM")


plt.show()



plt.scatter(df.loc[df.stim=='right','trial_time'],df.loc[df.stim=='right','trial_heading'],s=1,c='b',label='right')
plt.scatter(df.loc[df.stim=='left','trial_time'],df.loc[df.stim=='left','trial_heading'],s=1,c='r',label='left')
plt.scatter(df.loc[df.stim=='static','trial_time'],df.loc[df.stim=='static','trial_heading'],s=1,c='gray',alpha=0.1,label='static')
plt.legend()

plt.show()


for col in df.columns:
    if col not in ['timestamp', 'epoch', 'trial_time',  'seq_counter', 'frame_counter','stim']:
        plt.scatter(df.loc[df.epoch==1,'trial_time'],df.loc[df.epoch==1,col])
        plt.title(col)
        plt.show()





df2 = df.loc[df['epoch']==1,:]
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



