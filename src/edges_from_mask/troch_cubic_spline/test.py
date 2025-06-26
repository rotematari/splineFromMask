import torch
import matplotlib.pyplot as plt
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

# ---- 1. Synthetic noisy data ----
# true function
def f(u): return torch.sin(2*u)
N = 15
t = torch.linspace(0.0, 1.0, N)
with torch.no_grad():
    P_true = f(t)
noise = 0.2 * torch.randn_like(P_true)
P_noisy = P_true + noise

# ---- 2. Make P_i learnable ----
P = torch.nn.Parameter(P_noisy.clone())

# ---- 3. Optimizer ----
opt = torch.optim.Adam([P], lr=1e-2)
λ = 1e-2   # smoothing weight

for epoch in range(2000):
    opt.zero_grad()

    # build spline from current P
    coeffs = natural_cubic_spline_coeffs(t, P.unsqueeze(-1))  # shape (N,4,1)
    spline = NaturalCubicSpline(coeffs)

    # data‐fit loss: at the knots
    fit = spline.evaluate(t).squeeze(-1)    # s(t_i)
    loss_data = torch.mean((fit - P_noisy)**2)

    # smoothness loss: approximate ∫[s''(u)]² du by sampling
    u = torch.linspace(0.0, 1.0, 200, requires_grad=True)
    y = spline.evaluate(u).squeeze(-1)
    # first derivative
    dy = torch.autograd.grad(y, u, torch.ones_like(y), create_graph=True)[0]
    # second derivative
    ddy = torch.autograd.grad(dy, u, torch.ones_like(dy), create_graph=True)[0]
    loss_smooth = torch.mean(ddy**2)

    loss = loss_data + λ * loss_smooth
    loss.backward()
    opt.step()

# ---- 4. Plot before/after ----
plt.figure(figsize=(6,4))
# original noisy
plt.scatter(t.numpy(), P_noisy.numpy(), color='gray', label='noisy data')
# true
ut = torch.linspace(0.0, 1.0, 200)
plt.plot(ut.numpy(), f(ut).numpy(), 'k--', label='ground truth')
# fitted spline
coeffs = natural_cubic_spline_coeffs(t, P.unsqueeze(-1).detach())
spl = NaturalCubicSpline(coeffs)
yf = spl.evaluate(ut).squeeze(-1)
plt.plot(ut.numpy(), yf.numpy(), 'r-', label='smoothing spline')
plt.legend()
plt.title(f"Smoothing Spline (λ={λ})")
plt.show()
