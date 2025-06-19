#!/usr/bin/env python3

import socket
import select
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

HOST = '127.0.0.1'
PORT = 42069
timeout_in_seconds = 1
line_buffer = ""
data_rows = []

# Setup plot
plt.ion()
fig, ax = plt.subplots()
sc, = ax.plot([], [], 'b-')  # Initial empty line
ax.set_xlim(-2, 2)  # Adjust as needed
ax.set_ylim(-2, 2)
ax.set_xlabel('posx')
ax.set_ylabel('posy')
ax.set_title('Live FicTrac PosX vs PosY')

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    sock.bind((HOST, PORT))
    sock.setblocking(0)

    x_vals, y_vals = [], []

    #while True:
    for i in range(2000):
        ready = select.select([sock], [], [], timeout_in_seconds)

        if ready[0]:
            new_data = sock.recv(1024)
            if not new_data:
                break

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

                    # Live update
                    posx = row[14]
                    posy = row[15]
                    x_vals.append(posx)
                    y_vals.append(posy)

                    sc.set_data(x_vals, y_vals)
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.001)

                except ValueError:
                    print("Conversion error")

        else:
            print("No data, exiting")
            break

# Finalize
plt.ioff()
plt.show()