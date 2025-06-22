import os
import cv2
import numpy as np

def compute_difference_images(folder_path):
    files = os.listdir(folder_path)

    # Create mappings for res and _res images
    res_images = {f[4:]: f for f in files if f.startswith("res_") and f.endswith(".png")}
    _res_images = {f[5:]: f for f in files if f.startswith("_res_") and f.endswith(".png")}

    for key in res_images.keys():
        if key in _res_images:
            res_path = os.path.join(folder_path, res_images[key])
            _res_path = os.path.join(folder_path, _res_images[key])
            out_path = os.path.join(folder_path, "__res_" + key)

            img1 = cv2.imread(res_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(_res_path, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                print(f"Skipping unreadable pair: {res_images[key]} and {_res_images[key]}")
                continue
            if img1.shape != img2.shape:
                print(f"Shape mismatch in pair: {res_images[key]} and {_res_images[key]}")
                continue

            diff = cv2.absdiff(img1, img2)
            diff = diff/np.max(np.max(diff)) * 255
            cv2.imwrite(out_path + ".png", diff)
            print(f"Saved difference: {os.path.basename(out_path)}")

# Example usage
folder = r'bspline_dataset'
compute_difference_images(folder)
