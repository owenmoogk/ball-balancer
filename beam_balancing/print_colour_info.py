import cv2
import numpy as np

def analyze_image_colors(image_path):
    # Load image (BGR by default)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Convert to RGB for intuitive ordering
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    total_pixels = h * w

    # Split channels
    r, g, b = cv2.split(rgb)

    # Compute stats
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    std_r, std_g, std_b = np.std(r), np.std(g), np.std(b)
    min_r, min_g, min_b = np.min(r), np.min(g), np.min(b)
    max_r, max_g, max_b = np.max(r), np.max(g), np.max(b)

    print(f"Image: {image_path}")
    print(f"Resolution: {w}x{h} ({total_pixels:,} pixels)")
    print("\n--- RGB Statistics ---")
    print(f"Mean: R={mean_r:.2f}, G={mean_g:.2f}, B={mean_b:.2f}")
    print(f"Std Dev: R={std_r:.2f}, G={std_g:.2f}, B={std_b:.2f}")
    print(f"Min: R={min_r}, G={min_g}, B={min_b}")
    print(f"Max: R={max_r}, G={max_g}, B={max_b}")

    return {
        "mean": (mean_r, mean_g, mean_b),
        "std": (std_r, std_g, std_b),
        "min": (min_r, min_g, min_b),
        "max": (max_r, max_g, max_b),
    }

# Example usage
if __name__ == "__main__":
    analyze_image_colors("red.jpg")



#R = [208.65 – 233.09]
# G = [100.59 – 157.91]
# B = [178.71 – 207.93]