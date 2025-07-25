from psychopy import monitors, visual, core
import numpy as np
from psychopy.tools.monitorunittools import deg2pix, pix2deg


# ---- PARAMETERS ----
static_dur      = 5       # seconds the dot stays still before motion
motion_dur      = 10       # seconds for the motion phase
post_static_dur = 5       # seconds the dot stays still after motion
fps             = 90       # frames per second
win_size        = (2400, 1080)
dot_radius_deg  = 16      # dot radius in degrees of visual angle
arc_deg         = 75       # total sweep (°)
vel_deg_s       = 75       # angular speed (° per second)

# ---- SET UP MONITOR FOR DEG↔PIX CONVERSION ----
mon = monitors.Monitor('myMonitor', width=7.32, distance=2)  
# (replace width & distance with your monitor’s physical width (cm) & viewing distance (cm))
mon.setSizePix(win_size)

# compute pixel values
dot_radius_px  = deg2pix(dot_radius_deg, mon)  # dot radius in pixels
half_arc_pix   = deg2pix(arc_deg/2, mon)  # half the total arc in pixels
speed_px_s     = deg2pix(vel_deg_s,mon)
dx             = speed_px_s / fps

# ---- SET UP WINDOW, DOT & TEXT ----
win = visual.Window(
    size=win_size,
    units="pix",
    monitor=mon,
    color=(173, 216, 230),
    fullscr=False
)

dot = visual.Circle(
    win=win,
    radius=dot_radius_px,
    fillColor=(0,0,139),
    lineColor=(0,0,139),
    units="pix"
)

pos_text = visual.TextStim(
    win=win,
    text='',
    pos=(0, -win_size[1]//2 + 20),
    color='black',
    units='pix'
)

win.recordFrameIntervals = True

# ---- STATIC PHASE BEFORE MOTION ----
for _ in range(int(static_dur * fps)):
    dot.pos    = (0, -100)
    dot.radius = dot_radius_px
    pos_text.text = "x = 0.00"
    dot.draw()#; pos_text.draw()
    win.flip(); win.getMovieFrame()

# ---- MOTION PHASE (angular sweep at vel_deg_s) ----
n_motion = int(motion_dur * fps)
x        = 0         # start at center
direction = 1        # 1 = moving right first

for _ in range(n_motion):
    # update position in pixels
    x += direction * dx

    # bounce at ±half_arc_pix
    if x >=  half_arc_pix:
        x =  half_arc_pix
        direction *= -1
    elif x <= -half_arc_pix:
        x = -half_arc_pix
        direction *= -1

    # optional growth (sinusoidal) you had before
    # norm_pos      = abs(x) / half_arc_pix            # 0 → 1
    # angle         = norm_pos * (np.pi / 2)
    # growth_factor = 1 + np.sin(angle)
    # dot.radius    = dot_radius_px * growth_factor

    # log and on-screen
    print(f"x = {x:.2f} px ({pix2deg(x, mon):.2f}°), radius = {dot.radius:.2f}px")
    pos_text.text = f"x = {pix2deg(x,mon):.1f}°"

    dot.pos = (x, -100)
    dot.draw()#; pos_text.draw()
    win.flip(); win.getMovieFrame()

# ---- STATIC PHASE AFTER MOTION ----
for _ in range(int(post_static_dur * fps)):
    dot.pos    = (0, -100)
    dot.radius = dot_radius_px
    pos_text.text = "x = 0.00"
    dot.draw()#; pos_text.draw()
    win.flip(); win.getMovieFrame()

# ---- SAVE MOVIE & CLEAN UP ----
output_fname = "dot_deg_sweep.mp4"
win.saveMovieFrames(output_fname, fps=fps, codec='libx264')
print("Saved video to", output_fname)

win.close(); core.quit()
