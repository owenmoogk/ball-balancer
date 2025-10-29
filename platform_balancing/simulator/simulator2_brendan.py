# 3-arm Stewart-like platform between two equilateral triangles.
# Base joints: hinges along base edges at their midpoints (axis = edge direction).
# Top joints: balls at the midpoints of the TOP triangle edges.
# Top centroid height h is fixed.
# Modes:
#  • Tilt: set roll, pitch; solver finds in-plane spin ψ to satisfy hinge planes; reports lengths.
#  • Lengths: set three actuator lengths; Gauss–Newton solves roll, pitch, ψ.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

EPS = 1e-9
RAD = np.pi/180.0

def unit(v):
    n = np.linalg.norm(v)
    return v if n < EPS else v/n

def Rx(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rodrigues(u, th):
    u = unit(u)
    K = np.array([[0, -u[2], u[1]],[u[2], 0, -u[0]],[-u[1], u[0], 0]])
    I = np.eye(3)
    return I + np.sin(th)*K + (1-np.cos(th))*(K@K)

def equilateral_vertices(side):
    # centroid at origin; vertices at angles 90°, 210°, 330° in local z=0
    Rcent = side/np.sqrt(3.0)
    ang = np.deg2rad([90.0, 210.0, 330.0])
    return np.c_[Rcent*np.cos(ang), Rcent*np.sin(ang), np.zeros(3)]

def edge_midpoints(V):
    return np.array([(V[0]+V[1])/2, (V[1]+V[2])/2, (V[2]+V[0])/2])

class Base:
    def __init__(self, side, yaw_deg=0.0):
        self.update(side, yaw_deg)
    def update(self, side, yaw_deg):
        self.side = side
        self.yaw  = yaw_deg*RAD
        V0 = equilateral_vertices(side)
        c,s = np.cos(self.yaw), np.sin(self.yaw)
        Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        self.V = V0 @ Rz.T
        self.M = edge_midpoints(self.V)
        self.u = np.array([ unit(self.V[(i+1)%3] - self.V[i]) for i in range(3) ])

class TopPose:
    def __init__(self, side_top, h):
        self.side = side_top
        self.h = h
        self.alpha = 0.0
        self.beta  = 0.0
        self.psi   = 0.0
        self.V0 = equilateral_vertices(side_top)
    def set_height(self, h): self.h = h
    def set_side(self, s):
        self.side = s
        self.V0 = equilateral_vertices(s)
    def set_angles(self, alpha_deg, beta_deg, psi_deg=None):
        self.alpha = alpha_deg*RAD
        self.beta  = beta_deg*RAD
        if psi_deg is not None:
            self.psi = psi_deg*RAD
    def R_tilt(self): return Ry(self.beta) @ Rx(self.alpha)
    def R_total(self):
        Rt = self.R_tilt()
        n  = Rt @ np.array([0,0,1.0])
        Rpsi = rodrigues(n, self.psi)
        return Rpsi @ Rt
    def vertices(self):
        R = self.R_total()
        C = np.array([0.0,0.0,self.h])
        return self.V0 @ R.T + C
    def midpoints(self):
        return edge_midpoints(self.vertices())

def hinge_residual_sq(base: Base, Mp):
    # Hinge constraint: arm direction must be orthogonal to hinge axis u_i
    # => u_i · (Mp_i - M_i) = 0
    r = 0.0
    for i in range(3):
        r += float(np.dot(base.u[i], Mp[i]-base.M[i]))**2
    return r

def solve_spin_for_hinges(base: Base, pose: TopPose):
    # 1D golden-section search over ψ
    a,b = -np.pi, np.pi
    gr = (np.sqrt(5)-1)/2
    c = b - gr*(b-a); d = a + gr*(b-a)
    def f(psi):
        pose.psi = psi
        return hinge_residual_sq(base, pose.midpoints())
    fc, fd = f(c), f(d)
    for _ in range(64):
        if fc > fd:
            a = c; c = d; fc = fd; d = a + gr*(b-a); fd = f(d)
        else:
            b = d; d = c; fd = fc; c = b - gr*(b-a); fc = f(c)
    pose.psi = (a+b)/2

def lengths_from_pose(base: Base, pose: TopPose):
    return np.linalg.norm(pose.midpoints() - base.M, axis=1)

def solve_angles_from_lengths(base: Base, pose: TopPose, L_target, iters=30, lam=1e-3):
    # Gauss–Newton on p=[α,β,ψ]
    p = np.array([pose.alpha, pose.beta, pose.psi])
    def residuals(p):
        pose.alpha, pose.beta, pose.psi = p
        return lengths_from_pose(base, pose) - L_target
    r = residuals(p)
    for _ in range(iters):
        J = np.zeros((3,3))
        for j in range(3):
            dp = np.zeros(3); dp[j] = 1e-5
            r2 = residuals(p+dp)
            J[:,j] = (r2 - r)/1e-5
        A = J.T@J + lam*np.eye(3)
        g = J.T@r
        try: dp = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError: break
        if np.linalg.norm(dp) < 1e-6: break
        p = p + dp; r = residuals(p)
    pose.alpha, pose.beta, pose.psi = p

# --------- Plot/UI ----------
plt.close('all')
fig = plt.figure(figsize=(9.6, 8.2))
ax  = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
plt.subplots_adjust(left=0.08, bottom=0.36, right=0.98, top=0.92)

BASE_SIDE0 = 3.0
TOP_SIDE0  = 3.0
BASE_YAW0  = 0.0
H0         = 3.0
ROLL0      = 0.0
PITCH0     = 0.0

base = Base(BASE_SIDE0, BASE_YAW0)
pose = TopPose(TOP_SIDE0, H0)
pose.set_angles(ROLL0, PITCH0, 0.0)
solve_spin_for_hinges(base, pose)
L0 = lengths_from_pose(base, pose)

specs = [
    ("BaseSide", 0.5, 8.0, BASE_SIDE0),
    ("TopSide",  0.5, 8.0, TOP_SIDE0),
    ("Baseφ°",  -180, 180, BASE_YAW0),
    ("h",         0.2, 10.0, H0),
    ("Roll°",     -45,  45,  ROLL0),
    ("Pitch°",    -45,  45,  PITCH0),
    ("L1",        0.1, 10.0, L0[0]),
    ("L2",        0.1, 10.0, L0[1]),
    ("L3",        0.1, 10.0, L0[2]),
]

slider_axes = []
y0, h, pad = 0.32, 0.03, 0.006
for i in range(len(specs)):
    y = y0 - i*(h+pad)
    slider_axes.append(plt.axes([0.16, y, 0.72, h], facecolor='lightgoldenrodyellow'))

sliders = [Slider(ax=sa, label=lab, valmin=vmin, valmax=vmax, valinit=vin)
           for sa,(lab,vmin,vmax,vin) in zip(slider_axes, specs)]

ax_mode = plt.axes([0.02, 0.32, 0.10, 0.10])
mode_radio = RadioButtons(ax_mode, ("Tilt", "Lengths"), active=0)

edge_colors = ['tab:orange','tab:blue','tab:green']
base_edges = [ax.plot([], [], [], ls=':', lw=2,   color=edge_colors[i])[0] for i in range(3)]
axis_lines = [ax.plot([], [], [], ls='--', lw=1.4, color=edge_colors[i])[0] for i in range(3)]
arm_lines  = [ax.plot([], [], [], lw=3, marker='o', color='k')[0] for _ in range(3)]
edge_top   = [ax.plot([], [], [], lw=2.3, color=edge_colors[i])[0] for i in range(3)]
face_top   = None
text_box   = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left', fontsize=10)

def draw_top_face(Vp):
    global face_top
    if face_top is not None:
        try: face_top.remove()
        except Exception: pass
    face_top = ax.add_collection3d(Poly3DCollection([Vp], alpha=0.25, linewidths=0))

def current_mode(): return mode_radio.value_selected

def update(_=None):
    base_side   = sliders[0].val
    top_side    = sliders[1].val
    base_yawdeg = sliders[2].val
    hcen        = sliders[3].val

    base.update(base_side, base_yawdeg)
    pose.set_side(top_side)
    pose.set_height(hcen)

    if current_mode() == "Tilt":
        roll_deg  = sliders[4].val
        pitch_deg = sliders[5].val
        pose.set_angles(roll_deg, pitch_deg)  # keep ψ
        solve_spin_for_hinges(base, pose)
        L = lengths_from_pose(base, pose)
    else:
        Ldes = np.array([sliders[6].val, sliders[7].val, sliders[8].val])
        solve_spin_for_hinges(base, pose)
        solve_angles_from_lengths(base, pose, Ldes)
        L = lengths_from_pose(base, pose)

    V  = base.V
    M  = base.M
    u  = base.u
    Vp = pose.vertices()
    Mp = edge_midpoints(Vp)

    for i in range(3):
        j = (i+1)%3
        base_edges[i].set_data([V[i,0], V[j,0]], [V[i,1], V[j,1]])
        base_edges[i].set_3d_properties([0.0, 0.0])
        t = np.linspace(-base_side, base_side, 2)
        A = M[i] + np.outer(t, u[i])
        axis_lines[i].set_data(A[:,0], A[:,1])
        axis_lines[i].set_3d_properties(A[:,2])

    draw_top_face(Vp)
    for i in range(3):
        j = (i+1)%3
        edge_top[i].set_data([Vp[i,0], Vp[j,0]], [Vp[i,1], Vp[j,1]])
        edge_top[i].set_3d_properties([Vp[i,2], Vp[j,2]])

    for i in range(3):
        xs = [M[i,0], Mp[i,0]]
        ys = [M[i,1], Mp[i,1]]
        zs = [M[i,2], Mp[i,2]]
        arm_lines[i].set_data(xs, ys)
        arm_lines[i].set_3d_properties(zs)

    span = max(base_side, top_side)
    lim = max(span, np.max(L) + 0.5*span) + 0.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0.0, max(lim, hcen+span))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    txt = [
        f"Mode: {current_mode()}",
        f"h = {pose.h:.3f}, BaseSide = {base_side:.3f}, TopSide = {top_side:.3f}",
        f"roll = {pose.alpha/RAD:.2f}°, pitch = {pose.beta/RAD:.2f}°, spin ψ = {pose.psi/RAD:.2f}°",
        f"L = [{L[0]:.3f}, {L[1]:.3f}, {L[2]:.3f}]",
        f"hinge residual = {hinge_residual_sq(base, Mp):.2e}",
    ]
    text_box.set_text("\n".join(txt))
    ax.set_title('3-arm platform: base hinges along edges, top balls at edge midpoints')
    fig.canvas.draw_idle()

for s in sliders: s.on_changed(update)
mode_radio.on_clicked(update)

update()
plt.show()