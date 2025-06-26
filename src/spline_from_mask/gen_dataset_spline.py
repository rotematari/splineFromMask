import os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
def gen_random_bspline(k=3, s=0.1, num_ctrl_pts=None, num_ctrl_range=(4, 10), num_output_pts=200, dim=2, dense_pts_for_mask=1000):
    """
    Generate a random B-spline and sample points evenly distributed along arc length.
    
    Parameters
    ----------
    k : int, default 3
        Degree of the B-spline
    s : float, default 0.1
        Smoothing factor for spline fitting
    num_ctrl_pts : int or None
        Exact number of control points. If None, uses random number from num_ctrl_range
    num_ctrl_range : tuple, default (4, 10)
        Range for random number of control points (used only if num_ctrl_pts is None)
    num_output_pts : int, default 200
        Number of points to sample evenly along arc length for output
    dim : int, default 2
        Dimensionality (2D or 3D)
    dense_pts_for_mask : int, default 1000
        Number of dense points to generate for smooth mask rendering
    
    Returns
    -------
    spline_points : ndarray
        Points evenly distributed along arc length, shape (num_output_pts, dim)
    dense_spline_points : ndarray
        Dense points for mask generation, shape (dense_pts_for_mask, dim)
    c_norm : ndarray
        Normalized spline coefficients
    t : ndarray
        Knot vector
    k : int
        Spline degree
    """
    # 1) Determine number of control points
    if num_ctrl_pts is None:
        n_ctrl = np.random.randint(*num_ctrl_range)
    else:
        n_ctrl = num_ctrl_pts
    
    # 2) Generate random control points in [0,1]^dim
    ctrl_pts = np.random.rand(n_ctrl, dim).astype(np.float32)

    # 3) Fit B-spline to control points
    tck, u = splprep(ctrl_pts.T, s=s, k=k)
    t, c, k = tck

    # 4) Normalize each coefficient array to [0,1]
    c_norm = []
    for dim_vals in c:
        mn, mx = dim_vals.min(), dim_vals.max()
        if mx > mn:
            c_norm.append((dim_vals - mn) / (mx - mn))
        else:
            c_norm.append(np.zeros_like(dim_vals))
    c_norm = np.stack(c_norm, axis=0).astype(np.float32)
    tck_norm = (t, c_norm, k)

    # 5) Generate dense points for smooth mask rendering
    u_dense = np.linspace(0, 1, dense_pts_for_mask)
    dense_spline_points = np.vstack(splev(u_dense, tck_norm)).T  # (dense_pts_for_mask, dim)

    # 6) Sample points evenly distributed along arc length for dataset
    spline_points = resample_spline_by_arc_length(tck_norm, num_output_pts)
    
    # Validate that points are in [0,1] range
    if spline_points.any() < 0.0 or spline_points.any() > 1.0:
        raise ValueError("Spline points are not in [0,1] range!")
    if dense_spline_points.any() < 0.0 or dense_spline_points.any() > 1.0:
        raise ValueError("Dense spline points are not in [0,1] range!")
    
    return spline_points, dense_spline_points, c_norm, t, k
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
        projection_params=None,
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
    projection_params : dict or None
        If provided, reuse existing projection parameters (e1, e2, centroid, scale).
        If None, generate new random projection.

    Returns
    -------
    mask : (H,W) uint8 ndarray
        Binary image with the projected spline.
    projected_coords : (N,2) ndarray
        Projected coordinates in pixel space.
    projection_params : dict
        Projection parameters for reuse (e1, e2, centroid, scale).
    """
    W, H = img_size
    border = int(border_ratio * min(W, H))
    
    if projection_params is None:
        # Generate new random projection
        rng = np.random.default_rng(seed)
        
        # --- 1. random plane ----------------------------------------------------
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)          # random unit normal

        tmp = np.array([0, 0, 1]) if abs(n[2]) < 0.9 else np.array([0, 1, 0])
        e1 = np.cross(n, tmp);  e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)            # e2 ⟂ e1, n
        
        # Store projection parameters
        projection_params = {
            'e1': e1,
            'e2': e2,
            'centroid': spline.mean(axis=0),
            'scale': None  # Will be computed below
        }
    
    # Use projection parameters
    e1 = projection_params['e1']
    e2 = projection_params['e2']
    centroid = projection_params['centroid']

    # --- 2. project points --------------------------------------------------
    centered = spline - centroid     # put centroid at origin
    u = centered @ e1
    v = centered @ e2

    # --- 3. normalise to [0,1] with padding ---------------------------------
    if projection_params['scale'] is None:
        # Compute scale from current data
        ru, rv = u.ptp(), v.ptp()       # ptp = max - min
        scale = max(ru, rv) + 1e-9      # keep aspect ratio
        projection_params['scale'] = scale
    else:
        scale = projection_params['scale']
    
    u_norm = (u - u.mean()) / scale 
    v_norm = (v - v.mean()) / scale 

    px = (u_norm * (W - 1 - 2*border) + border).astype(int)
    py = ((1 - v_norm) * (H - 1 - 2*border) + border).astype(int)

    # --- 4. rasterise -------------------------------------------------------
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).line(list(zip(px, py)), fill=255, width=line_width)
    
    projected_coords = np.vstack((px, py)).T
    return np.asarray(img, dtype=np.uint8), projected_coords, projection_params

def build_dataset(base_dir, N=1000, img_size=(256,256),
                  num_ctrl_pts=None, num_ctrl_range=(4, 10), num_output_pts=20, 
                  order=3, s=0.1, dim=3, dense_pts_for_mask=1000):
    """
    Build a dataset of spline masks.
    
    Parameters
    ----------
    base_dir : str
        Base directory where the dataset will be created
    N : int
        Number of different splines to generate
    img_size : tuple
        Image size (width, height)
    num_ctrl_pts : int or None
        Exact number of control points for B-spline generation. If None, uses random from num_ctrl_range
    num_ctrl_range : tuple
        Range for random number of control points (used only if num_ctrl_pts is None)
    num_output_pts : int
        Number of points to sample evenly along arc length
    order : int
        B-spline degree
    s : float
        Smoothing factor
    dim : int
        Dimensionality (2D or 3D)
    dense_pts_for_mask : int
        Number of dense points for smooth mask rendering
    """
    # Generate dataset name from parameters
    W, H = img_size
    ctrl_str = f"{num_ctrl_pts}ctrl" if num_ctrl_pts is not None else f"{num_ctrl_range[0]}-{num_ctrl_range[1]}ctrl"
    s_str = f"s{s}".replace(".", "p")  # Replace decimal point with 'p'
    
    dataset_name = f"ds_{W}x{H}_{N}splines_{num_output_pts}pts_{ctrl_str}_k{order}_{s_str}_dim{dim}"
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    root = os.path.join(base_dir, dataset_name)
    
    print(f"Creating dataset: {dataset_name}")
    print(f"Full path: {root}")
    
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    counter = 0
    
    for i in range(N):
        # 1) Generate one random spline with both arc-length and dense sampling
        spline_points, dense_spline_points, c_norm, t, k = gen_random_bspline(
            k=order, s=s, 
            num_ctrl_pts=num_ctrl_pts, 
            num_ctrl_range=num_ctrl_range, 
            num_output_pts=num_output_pts, 
            dim=dim,
            dense_pts_for_mask=dense_pts_for_mask
        )

        # 2) apply a random rotation
        # R_mat = random_rotation_matrix()
        # rotated = spline3d @ R_mat.T

        line_width = np.random.randint(3, 6)  # Random line width between 1 and 5
        border_ratio = np.random.uniform(0.05, 0.3)  # Random border ratio between 0.05 and 0.3
        
        # 3) render to 2D mask using dense spline for smooth rendering
        if dim == 3:
            # First project dense points for mask generation
            mask, _, projection_params = project_to_random_plane_mask(
                dense_spline_points, img_size, line_width=line_width, border_ratio=border_ratio
            )
            # Then project arc-length sampled points using the same projection
            _, projected_spline_in_px, _ = project_to_random_plane_mask(
                spline_points, img_size, line_width=line_width, border_ratio=border_ratio, 
                projection_params=projection_params
            )
        elif dim == 2:
            # For 2D, generate mask from dense points
            mask, _ = spline2d_to_mask(
                dense_spline_points, img_size=img_size, line_width=line_width, 
                border_ratio=border_ratio, assume_unit_box=True, dtype=np.uint8
            )
            # Project the arc-length sampled points 
            _, projected_spline_in_px = spline2d_to_mask(
                spline_points, img_size=img_size, line_width=line_width, 
                border_ratio=border_ratio, assume_unit_box=True, dtype=np.uint8
            )
        import matplotlib.pyplot as plt
        # plt.imshow(mask, cmap='gray')
        # plt.plot(projected_spline_in_px[:, 0], projected_spline_in_px[:, 1], 'r-', linewidth=line_width)
        # plt.show()
        # 4) save with a simple index
        idx = f"{i:05d}"
        Image.fromarray((mask>0).astype(np.uint8)*255).save(
            os.path.join(img_dir, f"{idx}.png")
        )
        np.savez(
            os.path.join(lbl_dir, f"{idx}.npz"),
            knots=t.astype(np.float32),
            coeffs=c_norm,
            degree=np.int32(k),
            spline=projected_spline_in_px.astype(np.float32),  # projected spline in pixel coordinates
            mask=mask.astype(np.int32),  # binary mask
            line_width=np.int32(line_width)  # save line width
            
            
        )
        counter += 1
        if counter % 500 == 0:
            print(f"Saved {counter} samples...")

    print("Done!")

def resample_spline_by_arc_length(tck, num_points=200, initial_samples=1000):
    """
    Resample a B-spline so that points are evenly distributed along arc length.
    
    Parameters
    ----------
    tck : tuple
        B-spline representation (knots, coefficients, degree) from splprep
    num_points : int
        Number of points to sample along the arc length
    initial_samples : int
        Number of initial samples for arc length calculation (should be >> num_points)
    
    Returns
    -------
    resampled_points : ndarray
        Points evenly distributed along arc length, shape (num_points, dim)
    """
    t, c, k = tck
    
    # 1) Sample densely to compute arc length
    u_dense = np.linspace(0, 1, initial_samples)
    points_dense = np.vstack(splev(u_dense, tck)).T  # (initial_samples, dim)
    
    # 2) Calculate cumulative arc length
    segments = np.diff(points_dense, axis=0)  # (initial_samples-1, dim)
    segment_lengths = np.linalg.norm(segments, axis=1)  # (initial_samples-1,)
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])  # (initial_samples,)
    total_length = cumulative_lengths[-1]
    
    # 3) Create target arc lengths evenly spaced
    target_lengths = np.linspace(0, total_length, num_points)
    
    # 4) Interpolate to find u parameters corresponding to target arc lengths
    u_resampled = np.interp(target_lengths, cumulative_lengths, u_dense)
    
    # 5) Evaluate spline at the resampled u parameters
    resampled_points = np.vstack(splev(u_resampled, tck)).T  # (num_points, dim)

    return resampled_points

if __name__ == "__main__":
    build_dataset(base_dir="src/spline_dataset", 
                    N=32, num_output_pts=10, img_size=(256,256),
                    order=3, s=0.4, dim=2, num_ctrl_range=(4, 10),
                    dense_pts_for_mask=1000)