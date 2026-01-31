import torch
import matplotlib.pyplot as plt
from model import noise_predictor, SinusoidalEmbedding
from DDPM import scheduler
import numpy as np
@torch.no_grad()
def sample_ddpm(
    model,
    sampler,
    posenc,
    num_samples=1000,
    device='cpu'
):
    model.eval()
    model.to(device)
    
    # Move scheduler tensors to device
    sampler.alpha_prod = sampler.alpha_prod.to(device)
    sampler.one = sampler.one.to(device)
    sampler.timesteps = sampler.timesteps.to(device)
    
    # Move generator to device (or create new one on device)
    if device != 'cpu':
        sampler.generator = torch.Generator(device=device).manual_seed(42)
    else:
        sampler.generator = torch.Generator().manual_seed(42)

    # Number of dimensions (2D points)
    dim = 2

    # 1. Start from pure Gaussian noise (x_T)
    x = torch.randn(
        (num_samples, dim),
        generator=sampler.generator,
        device=device
    )

    # 2. Reverse diffusion loop
    for t in sampler.timesteps:
        t_batch = torch.full(
            (num_samples,),
            t.item(),
            dtype=torch.long,
            device=device
        )

        t_emb = posenc(t_batch)
        # Predict noise
        eps_pred = model(x, t_emb)

        # Denoise step
        x = sampler.step(t.item(), x, eps_pred)

    return x.cpu()

# --------------------------------------------------
# Load model and setup
# --------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load checkpoint
checkpoint = torch.load('dino_model_final.pth', map_location=device)
num_timestep = checkpoint.get('num_timestep', 700)  # Use saved timestep
print(num_timestep)
mean = checkpoint.get('mean', torch.tensor([0.0, 0.0]))  # Get normalization params
std = checkpoint.get('std', torch.tensor([1.0, 1.0]))

# Initialize model with SAME architecture as training
model = noise_predictor(inp_dim=2, embed_dim=64, hidden_dim=256, num_res_blocks=4)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # Ensure eval mode (disables dropout)

# Initialize scheduler with SAME num_timestep as training
sampler = scheduler(num_timestep=num_timestep)
posenc = SinusoidalEmbedding(dim=64)

# Generate samples - match the number of points after densification
# Original dino has ~142 points, densified with k=2 gives ~284 points
# But you can generate more for smoother visualization
num_samples = 200 # Match densified data, or use 700 for more points
samples = sample_ddpm(
    model,
    sampler,
    posenc,
    num_samples=num_samples,
    device=device
)

# CRITICAL: Denormalize the samples
# Convert mean and std to tensors if they're numpy arrays, and ensure they're on CPU
if isinstance(mean, np.ndarray):
    mean = torch.from_numpy(mean).float()
elif isinstance(mean, torch.Tensor):
    mean = mean.cpu().float()
else:
    mean = torch.tensor(mean).float()

if isinstance(std, np.ndarray):
    std = torch.from_numpy(std).float()
elif isinstance(std, torch.Tensor):
    std = std.cpu().float()
else:
    std = torch.tensor(std).float()

# Denormalize: x_original = x_normalized * std + mean
samples = samples * std + mean
# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(6, 6))
plt.scatter(
    samples[:, 0],
    samples[:, 1],
    s=10,
    alpha=0.7,
    c='blue',
    edgecolors='black',
    linewidths=0.5
)

plt.title("DDPM Generated Dino Shape")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True, alpha=0.3)

# Optional: Plot original for comparison
from dataloader import get_dataset
import numpy as np

# original_data = get_dataset("dino")
# plt.scatter(
#     original_data[:, 0],
#     original_data[:, 1],
#     s=8,
#     alpha=0.5,
#     c='red',
#     marker='x',
#     label='Original'
# )

plt.legend()
plt.tight_layout()
plt.show()