import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


df = pd.read_csv("./recording_fabian_1.csv")
df = df.replace("-", np.nan)
df[['px','py','pz']] = df["rrp_pos"].str.split(pat=',', expand=True).astype(float)
df[['qx','qy','qz','qw']] = df['rrp_quat'].str.split(pat=',', expand=True).astype(float)

# erstes gültiges Quaternion finden
valid_quat_idx = df[['qx','qy','qz','qw']].dropna().ne(0).any(axis=1).idxmax()
R0 = R.from_quat(df[['qx','qy','qz','qw']].iloc[valid_quat_idx].values)

size = 1.0
amp = 10

platform_pts = np.array([
    [-size, -size, 0],
    [ size, -size, 0],
    [ size,  size, 0],
    [-size,  size, 0]
])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(df['px'].min()-1, df['px'].max()+1)
ax.set_ylim(df['py'].min()-1, df['py'].max()+1)
ax.set_zlim(df['pz'].min()-1, df['pz'].max()+1)

platform_poly = Poly3DCollection([platform_pts], color='blue', alpha=0.5)
ax.add_collection3d(platform_poly)

# Tracers
trace_x = [[] for _ in range(4)]
trace_y = [[] for _ in range(4)]
trace_z = [[] for _ in range(4)]
colors = ['r', 'g', 'b', 'm']
trace_lines = []
for c in range(4):
    line, = ax.plot([], [], [], linestyle='--', color=colors[c], alpha=0.5)
    trace_lines.append(line)

ax.view_init(elev=30, azim=45)

def update(frame):
    pos = df[['px','py','pz']].iloc[frame].values
    pos = pos * amp
    quat = df[['qx','qy','qz','qw']].iloc[frame].values

    if np.any(np.isnan(pos)) or np.any(np.isnan(quat)) or np.all(quat==0):
        return [platform_poly] + trace_lines

    rot_abs = R.from_quat(quat)
    rot_delta = rot_abs * R0.inv()
    axis = rot_delta.as_rotvec()
    axis = axis * amp
    rot_amp = R.from_rotvec(axis)
    rot_final = rot_amp * R0
    Rm = rot_final.as_matrix()
    pts = (Rm @ platform_pts.T).T + pos
    platform_poly.set_verts([pts])

    for i in range(4):
        trace_x[i].append(pts[i,0])
        trace_y[i].append(pts[i,1])
        trace_z[i].append(pts[i,2])
        trace_lines[i].set_data(trace_x[i], trace_y[i])
        trace_lines[i].set_3d_properties(trace_z[i])

    return [platform_poly] + trace_lines

ani = FuncAnimation(fig, update, frames=len(df), interval=5, blit=False)
plt.show()
