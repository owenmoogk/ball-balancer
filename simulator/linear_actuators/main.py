import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# This import is necessary for 3D projections
from mpl_toolkits.mplot3d import Axes3D

# --- Define Platform and Base Geometry ---
BASE_HALF_WIDTH = 10.0  # Half the distance between base anchor points
PLATFORM_RADIUS = 8.0   # Radius of the platform (and half-distance of its anchors)

# Base anchor points (fixed on the ground at z=0)
B1 = np.array([-BASE_HALF_WIDTH, 0, 0])
B2 = np.array([BASE_HALF_WIDTH, 0, 0])
B3 = np.array([0, BASE_HALF_WIDTH, 0])

# Platform anchor points (in the platform's local coordinate system)
P1_local = np.array([-PLATFORM_RADIUS, 0, 0])
P2_local = np.array([PLATFORM_RADIUS, 0, 0])
P3_local = np.array([0, PLATFORM_RADIUS, 0])

# --- Helper function to draw the circular platform ---
def draw_platform(ax, center, radius, rotation_matrix):
    """
    Draws the circular platform in 3D space.
    """
    # Create points for a circle in the platform's local XY plane
    t = np.linspace(0, 2 * np.pi, 100)
    x_local = radius * np.cos(t)
    y_local = radius * np.sin(t)
    z_local = np.zeros_like(t)
    
    # Stack them into a 3x100 matrix
    local_coords = np.vstack((x_local, y_local, z_local))
    
    # Transform local coordinates to global coordinates
    # P_global = R * P_local + Center
    global_coords = rotation_matrix @ local_coords + center.reshape(3, 1)
    
    # Plot the circle
    ax.plot(global_coords[0, :], global_coords[1, :], global_coords[2, :], 'k')

# --- Main Update Function (Called by Sliders) ---
def update(val):
    """
    This function is called whenever a slider is moved.
    It recalculates and redraws the entire scene.
    """
    # Get current values from the sliders
    z = slider_z.val
    pitch_deg = slider_pitch.val
    pitch_rad = np.radians(pitch_deg)
    
    # --- Inverse Kinematics ---
    # We define the platform's pose (position and orientation)
    # and calculate the required link lengths and positions.
    
    # 1. Define platform center
    C = np.array([0, 0, z])
    
    # 2. Define platform orientation (Rotation matrix for pitch around Y-axis)
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    Ry = np.array([[c, 0, s],
                   [0, 1, 0],
                   [-s, 0, c]])
    
    # 3. Calculate global coordinates of platform anchor points
    P1_global = C + Ry @ P1_local
    P2_global = C + Ry @ P2_local
    P3_global = C + Ry @ P3_local
    
    # (Optional) Calculate the lengths of the two links
    L1 = np.linalg.norm(P1_global - B1)
    L2 = np.linalg.norm(P2_global - B2)
    L3 = np.linalg.norm(P3_global - B3)
    
    # --- Plotting ---
    ax.cla()  # Clear the previous drawing
    
    # Plot the base line
#     ax.plot([B1[0], B2[0]], [B1[1], B2[1]], [B1[2], B2[2]], 
#             'ko-', markersize=8, label='Base')
    
    # Plot the platform
    draw_platform(ax, C, PLATFORM_RADIUS, Ry)
    
    # Plot the two links (legs)
    ax.plot([B1[0], P1_global[0]], [B1[1], P1_global[1]], [B1[2], P1_global[2]], 
            'r-', linewidth=3, label=f'Link 1 (L={L1:.2f})')
    ax.plot([B2[0], P2_global[0]], [B2[1], P2_global[1]], [B2[2], P2_global[2]], 
            'b-', linewidth=3, label=f'Link 2 (L={L2:.2f})')
    ax.plot([B3[0], P3_global[0]], [B3[1], P3_global[1]], [B2[2], P3_global[2]], 
            'b-', linewidth=3, label=f'Link 3 (L={L3:.2f})')
            
    # Plot platform anchor points
    ax.plot([P1_global[0]], [P1_global[1]], [P1_global[2]], 'ro')
    ax.plot([P2_global[0]], [P2_global[1]], [P2_global[2]], 'bo')
    ax.plot([P3_global[0]], [P3_global[1]], [P3_global[2]], 'go')

    # Set plot limits and labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('2-DOF Platform Simulation')
    
    # Set fixed limits to prevent view from auto-scaling
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([0, 25])
    
    ax.legend()
    plt.draw()

# --- Setup the Figure and 3D Axis ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Adjust subplot position to make room for sliders
plt.subplots_adjust(bottom=0.25)

# --- Create Sliders ---
# Slider for Z height
ax_z = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_z = Slider(ax_z, 'Height (z)', 5.0, 20.0, valinit=10.0)

# Slider for Pitch angle
ax_pitch = plt.axes([0.25, 0.05, 0.65, 0.03])
slider_pitch = Slider(ax_pitch, 'Pitch (deg)', -30.0, 30.0, valinit=0.0)

# --- Connect Sliders to Update Function ---
slider_z.on_changed(update)
slider_pitch.on_changed(update)

# --- Initial Draw ---
update(0)

# --- Show the Plot ---
plt.show()