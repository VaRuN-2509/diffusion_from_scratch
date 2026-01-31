import torch
from dataloader import get_dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from DDPM import scheduler
from model import noise_predictor, SinusoidalEmbedding
import matplotlib.pyplot as plt

def process_data(input):

    if isinstance(input, np.ndarray):
        data = torch.from_numpy(input)
    else:
        data = input
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    # Add small epsilon to avoid division by zero
    std = std + 1e-8
    data = (data - mean) / std
    return data, mean, std

data = get_dataset("dino")

def densify(points, k=7):
    new_pts = []

    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]

        for j in range(k):
            alpha = j / k
            p = (1-alpha)*p1 + alpha*p2
            new_pts.append(p)

    return np.array(new_pts)

data = densify(data)


data, mean, std = process_data(data)

data = data + 0.01 * torch.randn_like(data)


plt.figure(figsize=(6,6))
plt.scatter(
    data[:,0],
    data[:,1],
)

plt.title("DDPM Generated Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()


sched = scheduler(num_timestep=1000)
posenc = SinusoidalEmbedding(dim=64)

# Use enhanced model with more capacity
model = noise_predictor(inp_dim=2, embed_dim=64, hidden_dim=256, num_res_blocks=4)
loss_fn = nn.MSELoss()

# Use lower learning rate with scheduling
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000, eta_min=1e-6)

# Training loop
batch_size = 16  # Increased batch size
num_epoch = 20000
N = data.shape[0]

best_loss = float('inf')
for i in range(num_epoch):
    idx = torch.randint(0, N, (batch_size,))
    data_in = data[idx]

    # t = torch.randint(0, sched.num_timestep, (batch_size,))
    u = torch.rand(batch_size)
    t = (u**2 * sched.num_timestep).long()


    x_t, noise = sched.add_noise(data_in, t)

    t_emb = posenc(t)
    noise_pred = model(x_t, t_emb) 


    loss = loss_fn(noise_pred,noise)
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler_lr.step()

    if i % 100 == 0:
        print(f"Epoch {i} | Loss: {loss.item():.6f} | LR: {scheduler_lr.get_last_lr()[0]:.6f}")
    
    # Save best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_timestep': sched.num_timestep,
            'mean': mean,
            'std': std,
        }, 'dino_model_best_loss.pth')
        if i % 1000 == 0:
            print(f"Saved best model with loss: {best_loss:.6f}")

print(f"Training complete! Best loss: {best_loss:.6f}")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_timestep': sched.num_timestep,
    'mean': mean,
    'std': std,
}, 'dino_model_final.pth')