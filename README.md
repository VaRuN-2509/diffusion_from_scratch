# Diffusion Models from Scratch

This project studies the behavior of diffusion models on low-dimensional geometric data, specifically **1D curves embedded in 2D space**, such as the Dino dataset.

Standard DDPMs are designed for full-dimensional distributions. When applied to thin manifolds, they often produce blurry or unstable samples. This project explores practical techniques to improve generation quality in such settings.

---

## 📌 Features

- Custom DDPM / DDIM sampler
- Sinusoidal timestep embeddings
- High-capacity MLP noise predictor
- Data densification and jittering
- x₀-prediction training objective
- Deterministic DDIM sampling
- Visualization of generated samples

---

## 📊 Dataset

- Dino (sparse 2D curve)
- Star (2D clustered shape)

The Dino dataset represents a challenging case of a 1D manifold embedded in 2D.

---

## ⚙️ Preprocessing

To stabilize training on thin manifolds:

- Densify points using interpolation
- Normalize to zero mean and unit variance
- Add small Gaussian noise (manifold thickening)

```python
data = densify(data, k=6)
data = normalize(data)
data += 0.003 * torch.randn_like(data)

```

---


## ▶️ Usage

python train.py

Trained models are saved as:

dino_model_best_loss.pth
dino_model_final.pth

For inference :

python inference.py

---

## 📚 References

Ho et al., DDPM (NeurIPS 2020)

Song et al., Score-Based Models (ICLR 2021)