import numpy as np
import cv2
from tqdm import tqdm

# Parameters
width, height = 2400, 1080
duration = 30  # seconds
fps = 90
total_frames = duration * fps

# Grating parameters
spatial_freq = 0.01  # cycles per pixel
temporal_freq = 0.05  # cycles per frame
angle = 0  # degrees

# Prepare coordinate grid
x = np.arange(width)
y = np.arange(height)
X, Y = np.meshgrid(x, y)

# Precompute spatial component of the grating
theta = np.deg2rad(angle)
X_theta = X * np.cos(theta) + Y * np.sin(theta)
spatial_term = 2 * np.pi * spatial_freq * X_theta

# Precompute constant static frame
static_grating = 0.5 + 0.5 * np.sin(spatial_term)
static_frame = (static_grating * 255).astype(np.uint8)
static_bgr = cv2.cvtColor(static_frame, cv2.COLOR_GRAY2BGR)

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'grating_tf{temporal_freq}_sf{spatial_freq}.mp4', fourcc, fps, (width, height))

# Main loop
for i in tqdm(range(total_frames), desc="Writing frames"):
    if i < 10 * fps:
        out.write(static_bgr)
    elif i < 20 * fps:
        phase = (i - 10 * fps) * temporal_freq * 2 * np.pi
        grating = 0.5 + 0.5 * np.sin(spatial_term + phase)
        frame = (grating * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    else:
        out.write(static_bgr)

out.release()
print("Done.")
