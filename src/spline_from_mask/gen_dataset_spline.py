import os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
def gen_random_bspline(k=3,s=0.1, num_ctrl_range=(4, 10), num_spline_pts=200, dim=2):
        # 1) random number of control points
    n_ctrl = np.random.randint(*num_ctrl_range)
    # 2) random 3D control points in [0,1]^3
    ctrl_pts = np.random.rand(n_ctrl, dim).astype(np.float32)

    # 3) fit natural cubic B-spline (no smoothing → interpolation)
    tck, u = splprep(ctrl_pts.T, s=s, k=k)
    t, c, k = tck

    # 4) normalize each coefficient array to [0,1]
    c_norm = []
    for dim_vals in c:
        mn, mx = dim_vals.min(), dim_vals.max()
        if mx > mn:
            c_norm.append((dim_vals - mn) / (mx - mn))
        else:
            c_norm.append(np.zeros_like(dim_vals))
    c_norm = np.stack(c_norm, axis=0).astype(np.float32)
    tck_norm = (t, c_norm, k)

    # 5) sample the normalized 3D spline
    u_fine = np.linspace(0, 1, num_spline_pts)
    spline = np.vstack(splev(u_fine, tck_norm)).T  # (num_spline_pts,dim)
    if spline.any() < 0.0 or spline.any() > 1.0:
        raise ValueError("Spline points are not in [0,1] range!")
    # print("Spline min max per dim:", [(dim.min(), dim.max()) for dim in spline3d.T])
    return spline , c_norm, t, k
def random_rotation_matrix():
    # uniform random rotation in SO(3)
    return R.random().as_matrix()
def spline2d_to_mask(
    pts,
    img_size=(256, 256),
    line_width=3,
    border_ratio=0.05,
    assume_unit_box=True,
    dtype=np.uint8,
):
    """
    Rasterise a 2-D spline/poly-line into a binary mask.

    Parameters
    ----------
    pts : (N,2) array_like
        Sequence of (x, y) points along the spline.
    img_size : (int,int), default (256,256)
        Output (width, height) of the mask.
    line_width : int, default 3
        Thickness of the drawn line in pixels.
    border_ratio : float, default 0.05
        Margin as a fraction of the smaller image side.
    assume_unit_box : bool, default True
        * True  –  points are already in [0,1] × [0,1].
        * False –  auto-scale/centre the points to fill the image.
    dtype : NumPy dtype, default np.uint8
        Desired dtype of the returned mask.

    Returns
    -------
    mask : (H,W) ndarray
        Binary image with 0 = background, 255 = spline.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be of shape (N,2)")

    W, H = img_size
    border = int(border_ratio * min(W, H))

    # ---------- 1. normalise to [0,1]² if necessary -------------------------
    if not assume_unit_box:
        # centre + uniform scale to keep aspect ratio
        span = pts.ptp(axis=0)             # [Δx, Δy]
        scale = span.max() + 1e-9          # avoid /0
        centred = pts - pts.mean(axis=0)
        pts_norm = centred / scale + 0.5   # shift to [0,1] roughly
    else:
        pts_norm = pts                     # already in unit box

    # ---------- 2. map to pixel coordinates --------------------------------
    px = (pts_norm[:, 0] * (W - 1 - 2 * border) + border).astype(int)
    py = ((1 - pts_norm[:, 1]) * (H - 1 - 2 * border) + border).astype(int)

    # clip to the image in case of numerical spill-over
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    # ---------- 3. rasterise -----------------------------------------------
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).line(list(zip(px, py)), fill=255, width=line_width)

    return np.asarray(img, dtype=dtype) , np.vstack((px, py)).T  # return projected spline in pixel coordinates
def project_to_random_plane_mask(
        spline,
        img_size=(256, 256),
        border_ratio=0.05,
        line_width=3,
        seed=None,
):
    """
    Project 3-D spline points onto a *random* 2-D plane and rasterise the result.

    Parameters
    ----------
    spline3d : (N,3) array_like
        Sequence of 3-D points along the spline.
    img_size : (int,int)
        Output image (width, height).
    border_ratio : float
        Fraction of image size reserved as an empty margin.
    line_width : int
        Thickness of the drawn line in pixels.
    seed : int or None
        Fix this for reproducible randomness.

    Returns
    -------
    mask : (H,W) uint8 ndarray
        Binary image with the projected spline.
    """
    W, H = img_size
    border = int(border_ratio * min(W, H))
    rng = np.random.default_rng(seed)

    # --- 1. random plane ----------------------------------------------------
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)          # random unit normal

    tmp = np.array([0, 0, 1]) if abs(n[2]) < 0.9 else np.array([0, 1, 0])
    e1 = np.cross(n, tmp);  e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)            # e2 ⟂ e1, n

    # --- 2. project points --------------------------------------------------
    centered = spline - spline.mean(axis=0)     # put centroid at origin
    u = centered @ e1
    v = centered @ e2

    # # --- 3. normalise to [0,1] with padding ---------------------------------
    ru, rv = u.ptp(), v.ptp()       # ptp = max - min
    scale = max(ru, rv) + 1e-9      # keep aspect ratio
    
    u_norm = (u - u.mean()) / scale 
    v_norm = (v - v.mean()) / scale 

    px = (u_norm * (W - 1 - 2*border) + border).astype(int)
    py = ((1 - v_norm) * (H - 1 - 2*border) + border).astype(int)

    # --- 4. rasterise -------------------------------------------------------
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).line(list(zip(px, py)), fill=255, width=line_width)
    return np.asarray(img, dtype=np.uint8) , np.vstack((px, py)).T  # return projected spline in pixel coordinates

def build_dataset(root, N=1000, img_size=(256,256),num_spline_pts=20, views_per_spline=10, order=3, s=0.1,dim=3):
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    counter = 0
    
    for i in range(N):
        # 1) sample one random 3D spline
        spline , c_norm, t, k = gen_random_bspline( num_spline_pts=num_spline_pts, k=order, s=s,dim = dim)

        for v in range(views_per_spline):
            # 2) apply a random rotation
            # R_mat = random_rotation_matrix()
            # rotated = spline3d @ R_mat.T

            line_width = np.random.randint(3, 6)  # Random line width between 1 and 5
            border_ratio = np.random.uniform(0.05, 0.3)  # Random border ratio between 0.05 and 0.3
            # 3) render to 2D mask
            if dim == 3:
                mask , projected_spline_in_px = project_to_random_plane_mask(spline, img_size,line_width=line_width,border_ratio=border_ratio)
            elif dim == 2:
                mask,projected_spline_in_px = spline2d_to_mask(spline, img_size=img_size, line_width=line_width, border_ratio=border_ratio, assume_unit_box=True, dtype=np.uint8)
            import matplotlib.pyplot as plt
            # plt.imshow(mask, cmap='gray')
            # plt.plot(projected_spline_in_px[:, 0], projected_spline_in_px[:, 1], 'r-', linewidth=line_width)
            # plt.show()
            # 4) save with a view-specific index
            idx = f"{i:05d}_{v:02d}"
            Image.fromarray((mask>0).astype(np.uint8)*255).save(
                os.path.join(img_dir, f"{idx}.png")
            )
            np.savez(
                os.path.join(lbl_dir, f"{idx}.npz"),
                knots=t.astype(np.float32),
                coeffs=c_norm,
                degree=np.int32(k),
                # spline=spline.astype(np.float32),  # original 3D points normelized to [0,1]
                spline=projected_spline_in_px.astype(np.float32),  # projected spline in pixel coordinates
                line_width=np.int32(line_width)  # save line width
                
            )
            counter += 1
            if counter % 500 == 0:
                print(f"Saved {counter} samples...")

    print("Done!")

if __name__ == "__main__":
    build_dataset(root="src/spline_from_mask/dataset_256_32_50pts", N=32,num_spline_pts=50, img_size=(256,256), views_per_spline=1,order=3, s=0.4,dim=2)

    # spline, c_norm, t, k = gen_random_bspline( num_spline_pts=200, k=3, s=0.1,dim = 3)
    
    # spline2d = spline[:, :2]  # take only the first two dimensions for 2D projection 
    # c_norm_2d = c_norm[:2, :]  # take only the first two dimensions for 2D coefficients
    
    # new_spline2d = np.vstack(splev(np.linspace(0, 1, 200), (t, c_norm_2d, k))).T  # sample the normalized 2D spline
    # print("2D Spline Points:\n", new_spline2d)
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # plt.plot(new_spline2d[:, 0], new_spline2d[:, 1], 'r-', label='Fitted 2D B-spline')
    # plt.plot(spline2d[:, 0], spline2d[:, 1], 'b', label='Original Points')

    # plt.legend()
    # plt.show()