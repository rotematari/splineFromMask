import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import os
from typing import Tuple, Literal

class BsplineMaskGenerator:
    def __init__(self,
                 image_size: int = 512,
                 curve_thickness: int = 7,
                 padding: float = 0.1,
                 num_ctrl_points: int = 6,
                 curve_samples: int = 1000,
                 grayscale_mode: Literal['linear', 'center'] = 'center',
                 seed: int = None):
        self.image_size = image_size
        self.curve_thickness = curve_thickness
        self.padding = padding
        self.num_ctrl_points = num_ctrl_points
        self.curve_samples = curve_samples
        self.grayscale_mode = grayscale_mode
        self.rng = np.random.default_rng(seed)

    def _generate_curve(self) -> np.ndarray:
        control_points = self.rng.random((2, self.num_ctrl_points))
        tck, _ = splprep(control_points, s=0, k=3)
        u = np.linspace(0, 1, self.curve_samples)
        spline = np.array(splev(u, tck))
        min_vals = np.min(spline, axis=1, keepdims=True)
        max_vals = np.max(spline, axis=1, keepdims=True)
        norm = (spline - min_vals) / (max_vals - min_vals + 1e-8)
        padded = norm * (1 - 2 * self.padding) + self.padding
        return padded.T

    def _resample_arclength(self, points: np.ndarray, num_samples: int) -> np.ndarray:
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
                t = (t_len - arc_lengths[idx - 1]) / (arc_lengths[idx] - arc_lengths[idx - 1] + 1e-8)
                pt = (1 - t) * points[idx - 1] + t * points[idx]
                resampled.append(pt)
        return np.array(resampled)

    def _draw_mask(self, curve: np.ndarray) -> np.ndarray:
        img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        scaled = (curve * (self.image_size - 1)).astype(np.int32)
        n = len(scaled)

        for i in range(1, n):
            pt1 = tuple(scaled[i - 1])
            pt2 = tuple(scaled[i])
            if self.grayscale_mode == 'linear':
                val = int(255 * i / (n - 1))
            elif self.grayscale_mode == 'center':
                progress = i / (n - 1)
                val = int(255 * (1 - abs(progress - 0.5) * 2))
            else:
                raise ValueError(f"Invalid grayscale_mode: {self.grayscale_mode}")
            cv2.line(img, pt1, pt2, color=val, thickness=self.curve_thickness)
        return img

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single (image, curve_points) sample."""
        curve = self._generate_curve()
        curve = self._resample_arclength(curve, self.curve_samples)
        mask = self._draw_mask(curve)
        return mask, curve

    def save_dataset(self, output_dir: str, count: int = 100, file_format: Literal['png', 'npy'] = 'png'):
        os.makedirs(output_dir, exist_ok=True)
        for i in range(count):
            mask, _ = self.generate()
            if file_format == 'png':
                cv2.imwrite(os.path.join(output_dir, f"mask_{i:05d}.png"), mask)
            elif file_format == 'npy':
                np.save(os.path.join(output_dir, f"mask_{i:05d}.npy"), mask)
