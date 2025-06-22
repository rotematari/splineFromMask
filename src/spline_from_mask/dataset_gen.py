import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

from spline_from_mask.bspline_image_label_gen import BsplineMaskGenerator


def generate_bspline_curve_padded(num_points=6, num_samples=1000, seed=None, padding=0.1):
    if seed is not None:
        np.random.seed(seed)
    control_points = np.random.rand(2, num_points)
    tck, _ = splprep(control_points, s=0, k=3)
    u = np.linspace(0, 1, num_samples)
    spline = np.array(splev(u, tck))
    min_vals = np.min(spline, axis=1, keepdims=True)
    max_vals = np.max(spline, axis=1, keepdims=True)
    normalized = (spline - min_vals) / (max_vals - min_vals + 1e-8)
    scale = 1 - 2 * padding
    padded = normalized * scale + padding
    return padded.T

def resample_curve_by_arclength(points, num_samples):
    deltas = np.diff(points, axis=0)
    distances = np.hypot(deltas[:, 0], deltas[:, 1])
    arc_lengths = np.concatenate([[0], np.cumsum(distances)])
    total_length = arc_lengths[-1]
    target_lengths = np.linspace(0, total_length, num_samples)
    resampled = []
    for t_len in target_lengths:
        idx = np.searchsorted(arc_lengths, t_len)
        if idx == 0 or idx >= len(points):
            resampled.append(points[min(idx, len(points)-1)])
        else:
            t = ((t_len - arc_lengths[idx - 1]) /
                 (arc_lengths[idx] - arc_lengths[idx - 1] + 1e-8))
            pt = (1 - t) * points[idx - 1] + t * points[idx]
            resampled.append(pt)
    return np.array(resampled)

def draw_grayscale_curve_symmetric(curve, image_size=512, thickness=7):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    scaled = (curve * (image_size - 1)).astype(np.int32)
    n = len(scaled)

    for i in range(1, n):
        pt1 = tuple(scaled[i - 1])
        pt2 = tuple(scaled[i])
        progress = i / (n - 1)
        grayscale_value = int(255 * (1 - abs(progress - 0.5) * 2))  # Brightest in the middle
        cv2.line(img, pt1, pt2, color=grayscale_value, thickness=thickness)

    return img

def draw_grayscale_curve(curve, image_size=512, thickness=7):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    scaled = (curve * (image_size - 1)).astype(np.int32)

    for i in range(1, len(scaled)):
        pt1 = tuple(scaled[i - 1])
        pt2 = tuple(scaled[i])
        grayscale_value = int(255 * i / (len(scaled) - 1))
        cv2.line(img, pt1, pt2, color=grayscale_value, thickness=thickness)

    return img
#
# # === Run It === example
# curve = generate_bspline_curve_padded(seed=42)
# resampled = resample_curve_by_arclength(curve, num_samples=1000)
# mask = draw_grayscale_curve_symmetric(resampled, thickness=7)
#
# plt.imshow(mask, cmap='gray')
# plt.title("Grayscale Mask Encoded by Arc-Length")
# plt.axis('off')
# plt.show()


gen = BsplineMaskGenerator(image_size=256, curve_thickness=3, grayscale_mode='linear', seed=42)

# On-the-fly sample
mask, curve = gen.generate()

# Visualize
import matplotlib.pyplot as plt
plt.imshow(mask, cmap='gray')
plt.title("Generated Mask")
plt.axis('off')
plt.show()

# Save dataset to disk

gen.save_dataset(output_dir="src/spline_from_mask/bspline_dataset", count=10000, file_format='png')

