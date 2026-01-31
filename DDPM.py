import torch
import numpy as np

class scheduler():
    def __init__(self,num_timestep,beta_start: float = 1e-4, beta_end: float = 5e-3):

        self.num_timestep = num_timestep
        self.betas = torch.linspace(beta_start**0.5,beta_end**0.5,num_timestep)**2
        self.alphas = 1-self.betas
        self.alpha_prod = torch.cumprod(self.alphas,dim=0)
        self.generator = torch.Generator().manual_seed(42)

        self.one = torch.tensor(1.0)

        self.timesteps = torch.from_numpy(np.arange(0, num_timestep)[::-1].copy())

        self.num_inference_steps = self.num_timestep
    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_timestep // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_timestep // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)
        
        # Get device from alpha_prod (will be moved in step function)
        alpha_prod = self.alpha_prod
        one = self.one

        alpha_prod_t = alpha_prod[timestep]
        alpha_prod_t_prev = alpha_prod[prev_t] if prev_t >= 0 else one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength=1):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    # def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
    #     t = timestep
    #     prev_t = self._get_previous_timestep(t)
        
    #     device = latents.device
        
    #     # Move tensors to correct device if needed
    #     alpha_prod = self.alpha_prod.to(device)
    #     one = self.one.to(device)

    #     # 1. compute alphas, betas
    #     alpha_prod_t = alpha_prod[t]
    #     alpha_prod_t_prev = alpha_prod[prev_t] if prev_t >= 0 else one
    #     beta_prod_t = 1 - alpha_prod_t
    #     beta_prod_t_prev = 1 - alpha_prod_t_prev
    #     current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    #     current_beta_t = 1 - current_alpha_t

    #     # 2. compute predicted original sample from predicted noise also called
    #     # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    #     pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    #     # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    #     # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    #     pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    #     current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    #     # 5. Compute predicted previous sample µ_t
    #     # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    #     pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

    #     # 6. Add noise
    #     variance = 0
    #     if t > 0:
    #         noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
    #         # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    #         variance_val = self._get_variance(t).to(device)
    #         variance = (variance_val ** 0.5) * noise
        
    #     # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
    #     # the variable "variance" is already multiplied by the noise N(0, 1)
    #     pred_prev_sample = pred_prev_sample + variance

    #     return pred_prev_sample
    
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):

        t = timestep
        prev_t = self._get_previous_timestep(t)

        device = latents.device

        alpha_prod = self.alpha_prod.to(device)
        one = self.one.to(device)

        # ᾱ_t and ᾱ_{t-1}
        alpha_t = alpha_prod[t]
        alpha_prev = alpha_prod[prev_t] if prev_t >= 0 else one

        # Predict x0
        x0_pred = (latents - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)

        # DDIM update (deterministic)
        dir_xt = torch.sqrt(1 - alpha_prev) * model_output
        prev_sample = torch.sqrt(alpha_prev) * x0_pred + dir_xt

        return prev_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:

        if not torch.is_tensor(original_samples):
            original_samples = torch.tensor(
                original_samples, dtype=torch.float32
            )

        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                timesteps, dtype=torch.long
            )

        
        alphas_cumprod = self.alpha_prod.to(device=original_samples.device, dtype=original_samples.dtype)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples,noise
