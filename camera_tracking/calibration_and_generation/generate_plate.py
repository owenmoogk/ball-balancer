# #!/usr/bin/env python3
# """
# Generate ring of AprilTags + printable chessboard sized to the same DPI/metric.
# Strict generator: only uses cv2.aruco.drawMarker. Fails loudly if drawMarker is
# absent or returns non-binary data.
# """
# import os
# import math
# import csv
# import yaml
# from PIL import Image, ImageDraw
# import numpy as np
# from reportlab.pdfgen import canvas
# from reportlab.lib.units import mm
# from reportlab.lib.utils import ImageReader
# import cv2

# # --------- user params (edit) ----------
# plate_radius_mm = 150
# tag_size_mm = 70
# num_tags = 6
# margin_from_edge_mm = 5
# dpi = 700
# rotate_to_face_center = True
# start_tag_id = 0
# output_dir = "generated"
# output_pdf = os.path.join(output_dir, "plate_layout.pdf")
# output_csv = os.path.join(output_dir, "tags_positions.csv")
# tag_png_folder = os.path.join(output_dir, "tag_pngs")
# # chessboard params (inner corners)
# CHESS_INNER_COLS = 9
# CHESS_INNER_ROWS = 6
# chess_square_mm = 30.0
# chess_output_png = os.path.join(output_dir, "chessboard.png")
# chess_output_pdf = os.path.join(output_dir, "chessboard.pdf")
# metadata_file = os.path.join(output_dir, "layout_metadata.yaml")
# # ---------------------------------------

# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(tag_png_folder, exist_ok=True)

# def mm_to_px(mm_val, dpi=dpi):
#     return int(round(mm_val * dpi / 25.4))

# def px_to_points(px, dpi=dpi):
#     return (px / dpi) * 72.0

# def ensure_tag_png(tag_id, size_px):
#     """
#     Strict create tag36h11_{id}.png using cv2.aruco.drawMarker only.
#     If drawMarker is unavailable or returns invalid data, raise RuntimeError
#     with diagnostics. Saves binary 'L' PNG at given size_px.
#     """
#     fname = os.path.join(tag_png_folder, f"tag36h11_{tag_id}.png")
#     if os.path.isfile(fname):
#         return fname

#     # verify aruco module
#     if not hasattr(cv2, "aruco"):
#         raise RuntimeError("cv2.aruco not present in this cv2 build.")

#     aruco = cv2.aruco

#     # get dictionary
#     try:
#         dict_obj = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
#     except Exception as e:
#         # try alternate getter name then error
#         try:
#             dict_obj = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
#         except Exception:
#             raise RuntimeError(f"Could not obtain APRILTAG_36h11 dictionary: {e}")

#     # Attempt pattern A: drawMarker(dict, id, sidePixels) -> returns ndarray
#     last_exc = None
#     try:
#         marker = aruco.drawMarker(dict_obj, int(tag_id), int(size_px))
#         if not isinstance(marker, np.ndarray):
#             raise RuntimeError("drawMarker returned non-ndarray type.")
#     except Exception as e:
#         last_exc = e
#         marker = None

#     # Attempt pattern B: drawMarker(dict, id, sidePixels, outArray, borderBits=1) -> in-place
#     if marker is None:
#         try:
#             out = np.zeros((size_px, size_px), dtype=np.uint8)
#             aruco.drawMarker(dict_obj, int(tag_id), int(size_px), out, 1)
#             # verify some non-zero pixels
#             if out.max() == 0:
#                 raise RuntimeError("drawMarker in-place call produced empty (all-zero) array.")
#             marker = out
#         except Exception as e:
#             last_exc = e
#             marker = None

#     # If still no marker, fail with diagnostics
#     if marker is None:
#         diag = {
#             "cv2_version": getattr(cv2, "__version__", "unknown"),
#             "cv2_file": getattr(cv2, "__file__", "unknown"),
#             "dict_obj_type": type(dict_obj).__name__ if 'dict_obj' in locals() else "None",
#             "last_exception": repr(last_exc)
#         }
#         raise RuntimeError(f"cv2.aruco.drawMarker not usable. Diagnostics: {diag}")

#     # At this point marker is an ndarray but may not be strict 0/255.
#     # Normalize to binary 0/255 robustly.
#     try:
#         m = np.array(marker)  # copy
#         if m.ndim == 3:
#             # some APIs return HxWx1 or HxWxC - reduce to single channel
#             m = m[...,0] if m.shape[2] == 1 else m[...,0]
#         # Values may be 0/1, 0/255, or small integers. Threshold to binary.
#         if m.max() <= 1:
#             m = (m * 255).astype(np.uint8)
#         elif m.max() == 255 and m.min() in (0,1):
#             # values are 0/255 or 0/1 mixed; threshold at 128
#             m = (m > 128).astype(np.uint8) * 255
#         else:
#             # generic threshold at mid-point
#             thresh = (int(m.max()) + int(m.min())) // 2
#             m = (m > thresh).astype(np.uint8) * 255

#         # Create PIL image as 8-bit 'L' and ensure hard edges by resizing with NEAREST.
#         pil = Image.fromarray(m.astype(np.uint8), mode="L")
#         if pil.size != (size_px, size_px):
#             pil = pil.resize((size_px, size_px), Image.NEAREST)
#         pil.save(fname, format="PNG", dpi=(dpi, dpi))
#         # final check read-back sanity
#         verify = np.array(Image.open(fname).convert("L"))
#         if not (verify.max() == 255 and verify.min() == 0):
#             raise RuntimeError("Saved PNG did not contain strict binary values (0/255).")
#         return fname
#     except Exception as e:
#         raise RuntimeError(f"Post-processing of marker failed: {repr(e)}")

# # ----------------- rest of generator -----------------
# plate_diam_mm = plate_radius_mm * 2
# page_w = plate_diam_mm * mm
# page_h = plate_diam_mm * mm
# c = canvas.Canvas(output_pdf, pagesize=(page_w, page_h))
# center_x = page_w / 2.0
# center_y = page_h / 2.0

# tag_px = mm_to_px(tag_size_mm, dpi=dpi)
# tag_pts = tag_size_mm * mm
# usable_radius_mm = plate_radius_mm - margin_from_edge_mm - (tag_size_mm/2.0)
# if usable_radius_mm <= 0:
#     raise ValueError("Tag size + margin too large for plate radius.")

# csv_rows = [("tag_id","x_mm","y_mm","rotation_deg")]

# for i in range(num_tags):
#     tag_id = start_tag_id + i
#     angle_rad = 2*math.pi * i / num_tags
#     x_mm = usable_radius_mm * math.cos(angle_rad)
#     y_mm = usable_radius_mm * math.sin(angle_rad)
#     rot_deg = (math.degrees(angle_rad) + 180.0) % 360.0 if rotate_to_face_center else 0.0

#     cx_pt = center_x + (x_mm * mm)
#     cy_pt = center_y + (y_mm * mm)
#     place_x = cx_pt - (tag_pts / 2.0)
#     place_y = cy_pt - (tag_pts / 2.0)

#     png_path = ensure_tag_png(tag_id, tag_px)
#     img = Image.open(png_path).convert("RGB")
#     img_reader = ImageReader(img)

#     c.saveState()
#     c.translate(cx_pt, cy_pt)
#     if rotate_to_face_center:
#         c.rotate(rot_deg)
#     c.drawImage(img_reader, -tag_pts/2.0, -tag_pts/2.0, width=tag_pts, height=tag_pts, mask='auto')
#     c.restoreState()

#     csv_rows.append((str(tag_id), f"{x_mm:.3f}", f"{y_mm:.3f}", f"{rot_deg:.3f}"))

# c.setLineWidth(1)
# c.circle(center_x, center_y, plate_radius_mm * mm, stroke=1, fill=0)
# c.showPage()
# c.save()

# with open(output_csv, "w", newline="") as f:
#     w = csv.writer(f)
#     w.writerows(csv_rows)

# print("Wrote:", output_pdf, output_csv)

# # --- chessboard ---
# chess_board_width_mm = (CHESS_INNER_COLS + 1) * chess_square_mm
# chess_board_height_mm = (CHESS_INNER_ROWS + 1) * chess_square_mm
# px_per_mm = dpi / 25.4
# width_px = max(200, int(round(chess_board_width_mm * px_per_mm)))
# height_px = max(200, int(round(chess_board_height_mm * px_per_mm)))
# square_px = int(round(chess_square_mm * px_per_mm))

# img = Image.new("RGB", (width_px, height_px), (255,255,255))
# draw = ImageDraw.Draw(img)
# for r in range(CHESS_INNER_ROWS + 1):
#     for ccol in range(CHESS_INNER_COLS + 1):
#         x0 = ccol * square_px
#         y0 = r * square_px
#         x1 = x0 + square_px
#         y1 = y0 + square_px
#         if (r + ccol) % 2 == 0:
#             draw.rectangle([x0, y0, x1, y1], fill=(0,0,0))

# img.save(chess_output_png, dpi=(dpi,dpi))
# img.save(chess_output_pdf, "PDF", resolution=dpi)
# print("Wrote:", chess_output_png, chess_output_pdf)

# # metadata
# meta = {
#     "dpi": int(dpi),
#     "tag_size_mm": float(tag_size_mm),
#     "plate_radius_mm": float(plate_radius_mm),
#     "num_tags": int(num_tags),
#     "chess_inner_cols": int(CHESS_INNER_COLS),
#     "chess_inner_rows": int(CHESS_INNER_ROWS),
#     "chess_square_mm": float(chess_square_mm),
#     "tag_png_folder": os.path.abspath(tag_png_folder),
#     "created_by": "generate_plate_strict.py"
# }
# with open(metadata_file, "w") as mf:
#     yaml.safe_dump(meta, mf)
# print("Wrote:", metadata_file)


#!/usr/bin/env python3


#!/usr/bin/env python3
"""
Generate a ring of AprilTags with printed dimensions, inner red & outer blue rings.
All dimension labels are placed outside the tags for readability.
"""
import os
import math
import csv
import yaml
from PIL import Image
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import cv2

from PIL import Image, ImageDraw
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import cv2


# ---------------- User Parameters ----------------
plate_radius_mm = 150
tag_size_mm = 75
num_tags = 4
margin_from_edge_mm = 30
dpi = 700
rotate_to_face_center = True
start_tag_id = 0
output_dir = "generated"
output_pdf = os.path.join(output_dir, "plate_layout.pdf")
output_csv = os.path.join(output_dir, "tags_positions.csv")
tag_png_folder = os.path.join(output_dir, "tag_pngs")
metadata_file = os.path.join(output_dir, "layout_metadata.yaml")

# Chessboard params (optional)
CHESS_INNER_COLS = 9
CHESS_INNER_ROWS = 6
chess_square_mm = 30.0
chess_output_png = os.path.join(output_dir, "chessboard.png")
chess_output_pdf = os.path.join(output_dir, "chessboard.pdf")
# --------------------------------------------------

os.makedirs(output_dir, exist_ok=True)
os.makedirs(tag_png_folder, exist_ok=True)

def mm_to_px(mm_val, dpi=dpi):
    return int(round(mm_val * dpi / 25.4))

def ensure_tag_png(tag_id, size_px):
    """Create tag36h11_{id}.png using cv2.aruco.drawMarker only."""
    fname = os.path.join(tag_png_folder, f"tag36h11_{tag_id}.png")
    if os.path.isfile(fname):
        return fname

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not present in this cv2 build.")
    aruco = cv2.aruco
    try:
        dict_obj = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    except Exception:
        dict_obj = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)

    marker = aruco.drawMarker(dict_obj, int(tag_id), int(size_px))
    m = marker if marker.max() > 1 else (marker * 255).astype(np.uint8)
    pil = Image.fromarray(m.astype(np.uint8), mode="L").resize((size_px, size_px), Image.NEAREST)
    pil.save(fname, format="PNG", dpi=(dpi, dpi))
    return fname

# ---------------- Canvas Setup ----------------
plate_diam_mm = plate_radius_mm * 2
page_w = plate_diam_mm * mm
page_h = plate_diam_mm * mm
c = canvas.Canvas(output_pdf, pagesize=(page_w, page_h))
center_x = page_w / 2.0
center_y = page_h / 2.0

tag_px = mm_to_px(tag_size_mm)
tag_pts = tag_size_mm * mm
usable_radius_mm = plate_radius_mm - margin_from_edge_mm - (tag_size_mm / 2.0)
if usable_radius_mm <= 0:
    raise ValueError("Tag size + margin too large for plate radius.")

# ---------------- Rings & Dimension Radii ----------------
inner_ring_radius_mm = usable_radius_mm - tag_size_mm / 2.0 - 5
tag_radius_mm = usable_radius_mm
outer_ring_radius_mm = usable_radius_mm + tag_size_mm / 2.0 + 5
ring_thickness_pt = 2.0 * mm

# --- Draw Rings ---
c.setLineWidth(ring_thickness_pt)
c.setStrokeColorRGB(1,0,0)  # Red inner
c.circle(center_x, center_y, inner_ring_radius_mm*mm, stroke=1, fill=0)
# c.setStrokeColorRGB(0,0,1)  # Blue outer
# c.circle(center_x, center_y, outer_ring_radius_mm*mm, stroke=1, fill=0)

# # --- Draw dimension lines & labels outside the tags ---
# # --- Enhanced dimensioning ---
# c.setLineWidth(1.5)
# label_offset_mm = 8
# tick_length_mm = 3

# # General grey radial lines with ticks
# c.setStrokeColorRGB(0.5,0.5,0.5)  # darker grey
# angles_deg = list(range(0, 360, 30))  # every 30 degrees
# for r_mm, label in zip([inner_ring_radius_mm, tag_radius_mm, outer_ring_radius_mm],
#                        ["Inner ring", "Tag center", "Outer ring"]):
#     for ang in angles_deg:
#         angle_rad = math.radians(ang)
#         x_end = center_x + r_mm * mm * math.cos(angle_rad)
#         y_end = center_y + r_mm * mm * math.sin(angle_rad)
#         # radial line
#         c.line(center_x, center_y, x_end, y_end)
#         # tick mark
#         tick_x0 = x_end - tick_length_mm/2*mm*math.sin(angle_rad)
#         tick_y0 = y_end + tick_length_mm/2*mm*math.cos(angle_rad)
#         tick_x1 = x_end + tick_length_mm/2*mm*math.sin(angle_rad)
#         tick_y1 = y_end - tick_length_mm/2*mm*math.cos(angle_rad)
#         c.line(tick_x0, tick_y0, tick_x1, tick_y1)
#         # label outside
#         x_label = center_x + (r_mm + label_offset_mm + tick_length_mm) * mm * math.cos(angle_rad)
#         y_label = center_y + (r_mm + label_offset_mm + tick_length_mm) * mm * math.sin(angle_rad)
#         c.setFont("Helvetica", 7)
#         c.drawCentredString(x_label, y_label, f"{label}={r_mm:.1f}mm")

# # X-axis (cyan) and Y-axis (magenta)
# c.setLineWidth(2)
# # X axis
# c.setStrokeColorRGB(0,1,1)  # cyan
# c.line(center_x - outer_ring_radius_mm*mm - 15*mm, center_y,
#        center_x + outer_ring_radius_mm*mm + 15*mm, center_y)
# c.setFont("Helvetica-Bold", 8)
# c.drawString(center_x + outer_ring_radius_mm*mm + 16*mm, center_y - 3*mm, "X")
# # Y axis
# c.setStrokeColorRGB(1,1,0)  # magenta
# c.line(center_x, center_y - outer_ring_radius_mm*mm - 15*mm,
#        center_x, center_y + outer_ring_radius_mm*mm + 15*mm)
# c.drawString(center_x + 1*mm, center_y + outer_ring_radius_mm*mm + 16*mm, "Y")

# # Add extra ticks on axes for each ring radius
# for r_mm, label in zip([inner_ring_radius_mm, tag_radius_mm, outer_ring_radius_mm],
#                        ["Inner", "Tag", "Outer"]):
#     # X-axis ticks
#     for sign in [-1,1]:
#         x_tick = center_x + sign*r_mm*mm
#         c.line(x_tick, center_y - tick_length_mm*mm, x_tick, center_y + tick_length_mm*mm)
#         c.setFont("Helvetica", 6)
#         c.drawCentredString(x_tick, center_y - 2*tick_length_mm*mm, f"{label}={r_mm:.1f}")
#     # Y-axis ticks
#     for sign in [-1,1]:
#         y_tick = center_y + sign*r_mm*mm
#         c.line(center_x - tick_length_mm*mm, y_tick, center_x + tick_length_mm*mm, y_tick)
#         c.drawCentredString(center_x - 2*tick_length_mm*mm, y_tick - 1*mm, f"{label}={r_mm:.1f}")


# ---------------- Place AprilTags ----------------
csv_rows = [("tag_id","x_mm","y_mm","rotation_deg")]
for i in range(num_tags):
    tag_id = start_tag_id + i
    angle_rad = 2 * math.pi * i / num_tags

    x_mm = usable_radius_mm * math.cos(angle_rad)
    y_mm = usable_radius_mm * math.sin(angle_rad)
    rot_deg = (math.degrees(angle_rad) + 180.0) % 360.0 if rotate_to_face_center else 0.0

    cx_pt = center_x + x_mm * mm
    cy_pt = center_y + y_mm * mm

    png_path = ensure_tag_png(tag_id, tag_px)
    img = Image.open(png_path).convert("RGB")
    img_reader = ImageReader(img)

    c.saveState()
    c.translate(cx_pt, cy_pt)
    if rotate_to_face_center:
        c.rotate(rot_deg)
    c.drawImage(img_reader, -tag_pts/2, -tag_pts/2, width=tag_pts, height=tag_pts, mask='auto')
    c.restoreState()

    csv_rows.append((str(tag_id), f"{x_mm:.3f}", f"{y_mm:.3f}", f"{rot_deg:.3f}"))

# ---------------- Outer Plate Boundary ----------------
c.setLineWidth(1)
c.setStrokeColorRGB(0,0,0)
c.circle(center_x, center_y, plate_radius_mm*mm, stroke=1, fill=0)

c.showPage()
c.save()

# ---------------- CSV Output ----------------
with open(output_csv, "w", newline="") as f:
    csv.writer(f).writerows(csv_rows)
print("Wrote:", output_pdf, output_csv)

# ---------------- Optional Chessboard ----------------
chess_board_width_mm = (CHESS_INNER_COLS + 1) * chess_square_mm
chess_board_height_mm = (CHESS_INNER_ROWS + 1) * chess_square_mm
px_per_mm = dpi / 25.4
width_px = max(200, int(round(chess_board_width_mm * px_per_mm)))
height_px = max(200, int(round(chess_board_height_mm * px_per_mm)))
square_px = int(round(chess_square_mm * px_per_mm))

img = Image.new("RGB", (width_px, height_px), (255,255,255))
draw = ImageDraw.Draw(img)
for r in range(CHESS_INNER_ROWS + 1):
    for ccol in range(CHESS_INNER_COLS + 1):
        x0 = ccol * square_px
        y0 = r * square_px
        x1 = x0 + square_px
        y1 = y0 + square_px
        if (r + ccol) % 2 == 0:
            draw.rectangle([x0, y0, x1, y1], fill=(0,0,0))

img.save(chess_output_png, dpi=(dpi,dpi))
img.save(chess_output_pdf, "PDF", resolution=dpi)
print("Wrote:", chess_output_png, chess_output_pdf)

# ---------------- Metadata ----------------
meta = {
    "dpi": dpi,
    "tag_size_mm": tag_size_mm,
    "plate_radius_mm": plate_radius_mm,
    "num_tags": num_tags,
    "chess_inner_cols": CHESS_INNER_COLS,
    "chess_inner_rows": CHESS_INNER_ROWS,
    "chess_square_mm": chess_square_mm,
    "tag_png_folder": os.path.abspath(tag_png_folder),
    "created_by": "generate_plate_strict_with_dimensions.py"
}
with open(metadata_file, "w") as mf:
    yaml.safe_dump(meta, mf)
print("Wrote:", metadata_file)
