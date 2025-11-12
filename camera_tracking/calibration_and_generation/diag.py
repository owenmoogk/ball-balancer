#!/usr/bin/env python3
# diag_aruco.py
import sys, os
import importlib, traceback
from pathlib import Path
from PIL import Image
import numpy as np

print("Python executable:", sys.executable)
print("Python version:", sys.version.splitlines()[0])

try:
    import cv2
    print("cv2 version:", cv2.__version__)
    try:
        print("cv2 file:", cv2.__file__)
    except Exception:
        pass
except Exception as e:
    print("Failed to import cv2:", e)
    sys.exit(1)

# show where pip would install / module resolution
try:
    import site
    print("site-packages dirs:", site.getsitepackages() if hasattr(site, 'getsitepackages') else site.getusersitepackages())
except Exception:
    pass

print("\nChecking cv2.aruco availability and attributes...\n")
aruco = None
try:
    aruco = cv2.aruco
    print("cv2.aruco found.")
    attrs = sorted([a for a in dir(aruco) if not a.startswith("_")])
    print("Some aruco attributes:", attrs[:40])
except Exception as e:
    print("cv2.aruco not available:", e)
    traceback.print_exc()

# Attempt to get dictionary object
dict_obj = None
if aruco is not None:
    for getter in ("getPredefinedDictionary", "Dictionary_get", "Dictionary_create"):
        try:
            func = getattr(aruco, getter)
            print(f"Found {getter}():", func)
            # Try common arg for APRILTAG_36h11
            try:
                dict_obj = func(cv2.aruco.DICT_APRILTAG_36h11)
            except Exception as e:
                # try without namespaced constant
                try:
                    dict_obj = func(aruco.DICT_APRILTAG_36h11)
                except Exception:
                    dict_obj = None
            if dict_obj is not None:
                print("Obtained dictionary object via", getter)
                break
        except Exception:
            continue
    if dict_obj is None:
        print("Could not obtain dictionary object for DICT_APRILTAG_36h11. Available constants (sample):")
        consts = [c for c in dir(aruco) if c.startswith("DICT_")]
        print(consts[:40])

# Try drawMarker patterns
out_dir = Path("generated/tag_pngs")
out_dir.mkdir(parents=True, exist_ok=True)
size_px = 236
sample_id = 0
success = False

if dict_obj is not None:
    print("\nTrying drawMarker patterns...\n")
    # pattern A: drawMarker(dict, id, sidePixels, img, borderBits=1) -> in-place
    try:
        marker = np.zeros((size_px, size_px), dtype=np.uint8)
        print("Attempting in-place drawMarker(dict, id, size, marker, borderBits=1)")
        aruco.drawMarker(dict_obj, int(sample_id), size_px, marker, 1)
        print("After call: marker.shape", marker.shape, "max", int(marker.max()))
        if marker.max() > 0:
            img = Image.fromarray(marker).convert("RGB")
            p = out_dir / f"test_inplace_{sample_id}.png"
            img.save(p, dpi=(300,300))
            print("Wrote", p)
            success = True
    except Exception:
        print("in-place drawMarker failed:")
        traceback.print_exc()

    # pattern B: drawMarker(dict, id, sidePixels) -> returns ndarray
    if not success:
        try:
            print("Attempting return drawMarker(dict, id, size)")
            marker_ret = aruco.drawMarker(dict_obj, int(sample_id), size_px)
            print("Returned type:", type(marker_ret))
            if isinstance(marker_ret, np.ndarray):
                print("marker_ret.shape", marker_ret.shape, "max", int(marker_ret.max()))
                if marker_ret.max() > 0:
                    img = Image.fromarray(marker_ret).convert("RGB")
                    p = out_dir / f"test_return_{sample_id}.png"
                    img.save(p, dpi=(300,300))
                    print("Wrote", p)
                    success = True
            else:
                print("drawMarker returned non-ndarray:", type(marker_ret))
        except Exception:
            print("return drawMarker failed:")
            traceback.print_exc()

if not success:
    print("\nNo marker image produced by cv2.aruco.drawMarker. Diagnostics summary and next steps:")
    print("1) Confirm you are running the same python interpreter used to install packages.")
    print("   Run: python3 -m pip show opencv-contrib-python")
    print("   Run: python3 -c \"import cv2; print(cv2.__version__, cv2.__file__)\"")
    print("2) Reinstall opencv-contrib-python cleanly using the same python:")
    print("   python3 -m pip uninstall -y opencv-python opencv-contrib-python")
    print("   python3 -m pip install --no-cache-dir opencv-contrib-python")
    print("   If macOS and issues persist try a specific version, e.g.:")
    print("   python3 -m pip install --no-cache-dir opencv-contrib-python==4.7.0.68")
    print("3) If you cannot install contrib in your environment, run the provided fallback I can supply (pure-Python tag renderer).")
    print("4) Paste the full output of this script here if you want me to interpret the exact failure.")
else:
    print("\nSuccess: a test PNG was written to", list(out_dir.iterdir()))
    print("Open that PNG. If it looks like a black/white AprilTag then re-run generate_plate.py.")