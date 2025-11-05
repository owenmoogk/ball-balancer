import cv2
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_circular_aruco_sheet(
    num_markers=6,
    marker_size_mm=30,
    platform_radius_mm=100,
    dpi=300,
    dictionary_name=cv2.aruco.DICT_4X4_50,
    output_path="aruco_circular_sheet.png",
    npz_output="aruco_marker_corners.npz",
    ensure_inside_cut=True,
    include_cut_circle=True,
):
    mm_per_inch = 25.4
    px_per_mm = dpi / mm_per_inch

    safe_radius_mm = platform_radius_mm
    if ensure_inside_cut:
        safe_radius_mm -= marker_size_mm * np.sqrt(2)/2

    marker_size_px = int(round(marker_size_mm * px_per_mm))
    platform_radius_px = int(round(safe_radius_mm * px_per_mm))
    margin_mm = marker_size_mm * 2
    canvas_side_mm = (platform_radius_mm + margin_mm) * 2
    canvas_side_px = int(round(canvas_side_mm * px_per_mm))

    sheet = Image.new("L", (canvas_side_px, canvas_side_px), 255)
    draw = ImageDraw.Draw(sheet)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_name)

    cx, cy = canvas_side_px // 2, canvas_side_px // 2
    all_corners = []
    all_ids = []
    all_centers = []

    for i in range(num_markers):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size_px)
        marker_img = Image.fromarray(marker_img)

        theta = 2 * math.pi * i / num_markers
        x_center = cx + platform_radius_px * math.cos(theta)
        y_center = cy + platform_radius_px * math.sin(theta)

        x0 = int(x_center - marker_size_px / 2)
        y0 = int(y_center - marker_size_px / 2)
        sheet.paste(marker_img, (x0, y0))

        idText = f"ID {i}"
        if i == 0:
            idText += " -> +X"
        draw.text((x0, y0 + marker_size_px + 5), idText, fill=0)
        
        # --- Compute 3D coordinates ---
        half = marker_size_mm / 2
        local_corners = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ])
        Rz = np.array([
            [ math.cos(theta + math.pi/2), -math.sin(theta + math.pi/2), 0],
            [ math.sin(theta + math.pi/2),  math.cos(theta + math.pi/2), 0],
            [ 0, 0, 1]
        ])
        center = np.array([
            safe_radius_mm * math.cos(theta),
            safe_radius_mm * math.sin(theta),
            0
        ])
        world_corners = (Rz @ local_corners.T).T + center
        all_corners.append(world_corners)
        all_ids.append(i)
        all_centers.append(center)

    # --- Cut circle ---
    if include_cut_circle:
        cut_r_px = int(platform_radius_mm * px_per_mm)
        draw.ellipse(
            [(cx - cut_r_px, cy - cut_r_px),
             (cx + cut_r_px, cy + cut_r_px)],
            outline=0, width=2
        )
        draw.text((cx + cut_r_px + 10, cy), "Cutting guide", fill=0)

    # --- Save outputs ---
    sheet.save(output_path, dpi=(dpi, dpi))
    np.savez(
        npz_output,
        marker_ids=np.array(all_ids),
        marker_corners=np.array(all_corners),
        marker_centers=np.array(all_centers),
        marker_size_mm=marker_size_mm,
        platform_radius_mm=platform_radius_mm,
        dpi=dpi,
    )

    print(f"✅ Saved layout to {output_path}")
    print(f"✅ Saved 3D corner data to {npz_output}")

if __name__ == "__main__":
    generate_circular_aruco_sheet(
        num_markers=6,
        marker_size_mm=30,
        platform_radius_mm=300,
        dpi=300,
        ensure_inside_cut=True,
    )
