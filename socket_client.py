#!/usr/bin/env python3

import socket
import select
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

HOST = '127.0.0.1'  # The (receiving) host IP address (sock_host)
PORT = 42069        # The (receiving) host port (sock_port)
timeout_in_seconds = 1
line_buffer = ""
data_rows = []


# TCP
# Open the connection (ctrl-c / ctrl-break to quit)
#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#    sock.connect((HOST, PORT))

# UDP
# Open the connection (ctrl-c / ctrl-break to quit)
timeoutcounter = 0
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.bind((HOST, PORT))
    sock.setblocking(0)

    #while True:
    while True:
        print("Waiting for data...")
        ready = select.select([sock], [], [], timeout_in_seconds)

        if ready[0]:
            new_data = sock.recv(1024)
            if not new_data:
                break
            elif new_data:
                timeoutcounter = 0 


            line_buffer += new_data.decode('UTF-8')

            while "\n" in line_buffer:
                endline = line_buffer.find("\n")
                line = line_buffer[:endline]
                line_buffer = line_buffer[endline + 1:]

                toks = line.split(", ")
                if len(toks) < 24 or toks[0] != "FT":
                    print("Bad read")
                    continue

                try:
                    row = np.array(toks[1:24], dtype=np.float64)
                    data_rows.append(row)
                except ValueError:
                    print("Conversion error")
    
        else:
            print('retrying...')
            timeoutcounter +=1
            print(f"Timeout counter: {timeoutcounter}")
            if timeoutcounter > 10:
                print("Timeout reached, exiting...")
                break


columns = [
    "cnt", "dr_cam_x", "dr_cam_y", "dr_cam_z", "err",
    "dr_lab_x", "dr_lab_y", "dr_lab_z",
    "r_cam_x", "r_cam_y", "r_cam_z",
    "r_lab_x", "r_lab_y", "r_lab_z",
    "posx", "posy", "heading",
    "step_dir", "step_mag", "intx", "inty",
    "ts", "seq"
]

column_names = [
    "frame_counter",
    "delta_rot_cam_x", "delta_rot_cam_y", "delta_rot_cam_z", "delta_rot_err",
    "delta_rot_lab_x", "delta_rot_lab_y", "delta_rot_lab_z",
    "abs_rot_cam_x", "abs_rot_cam_y", "abs_rot_cam_z",
    "abs_rot_lab_x", "abs_rot_lab_y", "abs_rot_lab_z",
    "posx", "posy", "heading",
    "movement_dir", "speed",
    "int_fwd", "int_side",
    "timestamp",
    "seq_counter"
]


# Use .vstack() to create array first (as before)
data_array = np.vstack(data_rows) if data_rows else np.empty((0, 23))

# Convert to DataFrame
df = pd.DataFrame(data_array, columns=column_names)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"fictrac_log_{timestamp}.csv", index=False)


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

x = df['posx'].values
y = df['posy'].values
t = df['frame_counter'].values-min(df['frame_counter'].values)
err = df['delta_rot_err']
seq = df['seq_counter'].values


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
