import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from spline_from_mask.datasets import SplineEndPointDataset

# 0) Load dataset & extract mask and endpoints
ds = SplineEndPointDataset(
    root='src/spline_from_mask/dataset_256_32_50pts',
    img_size=(256,256),
    num_pts=50,
    normalize=True
)

# mask_raw: torch.Tensor of shape (1, H, W) ─ squeeze to (H, W)
mask_raw = ds[0][0]
mask = mask_raw.squeeze(0).numpy()           # now shape (H, W)

# endpoints
(ep0, ep1) = ds[0][1]
y0, x0 = ep0
y1, x1 = ep1

# 1) Build a 2D distance map
inv_mask = 1 - (mask>0).astype(np.uint8)
distmap = distance_transform_edt(inv_mask).astype(np.float32)  # (H, W)

# 2) Pack as a 4D tensor for 2D grid_sample
#    shape = (N=1, C=1, H, W)
dist = torch.from_numpy(distmap)[None,None,:,:]

# 3) Initialize control‐points (N_ctrl × 2) by linear interpolation
N_ctrl = 8
P_init = np.linspace([x0,y0], [x1,y1], N_ctrl)  # (N_ctrl, 2)
P = torch.nn.Parameter(torch.from_numpy(P_init).float())

def clamp_ends(param):
    with torch.no_grad():
        param[0].copy_(torch.tensor([x0,y0]))
        param[-1].copy_(torch.tensor([x1,y1]))

# 4) Knot vector via chord‐length
d = np.linalg.norm(np.diff(P_init,axis=0), axis=1)
s = np.concatenate(([0.], np.cumsum(d)))
t_knots = torch.from_numpy(s).float()          # (N_ctrl,)

# 5) Optimize
opt = torch.optim.Adam([P], lr=1e-2)
M = 100
u = torch.linspace(t_knots[0], t_knots[-1], M)

for _ in range(500):
    opt.zero_grad()
    clamp_ends(P)

    # a) Fit spline to P
    coeffs = natural_cubic_spline_coeffs(t_knots, P)   # (N_ctrl,4,2)
    spline = NaturalCubicSpline(coeffs)
    pts   = spline.evaluate(u).squeeze(-1)             # (M,2) [x,y]

    # b) Build a 4D grid for 2D sampling
    xs = pts[:,0].clamp(0, mask.shape[1]-1)
    ys = pts[:,1].clamp(0, mask.shape[0]-1)
    grid = torch.stack([
        2*xs/(mask.shape[1]-1) - 1,
        2*ys/(mask.shape[0]-1) - 1
    ], dim=-1)                  # (M,2)
    grid = grid.unsqueeze(0).unsqueeze(2)  # → (1, M, 1, 2)

    # c) Sample distance‐map
    sampled = F.grid_sample(
        dist, grid,
        mode='bilinear',
        align_corners=True
    )                                 
    D = sampled[0,0,:,0]              # (M,)

    loss_mask   = D.pow(2).mean()
    # d) Smoothness penalty
    diffs        = P[1:] - P[:-1]
    loss_smooth = (diffs[1:] - diffs[:-1]).norm(dim=1).pow(2).mean()

    (loss_mask + 0.1*loss_smooth).backward()
    opt.step()

clamp_ends(P)

# 6) Extract final 100‐point curve
coeffs = natural_cubic_spline_coeffs(t_knots, P.detach())
curve  = NaturalCubicSpline(coeffs).evaluate(u).squeeze(-1).cpu().numpy()  # (M,2)

# 7) Visualize
plt.imshow(mask, cmap='gray')
plt.plot(curve[:,0], curve[:,1], 'r-', lw=2)
plt.scatter([x0,x1],[y0,y1], c='yellow', s=50, marker='x')
plt.axis('off')
plt.show()
