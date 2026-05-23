import streamlit as st
import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# 1. MODEL ARCHITECTURE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args  = t[:, None].float() * freqs[None, :]
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        t_out = self.time_mlp(t_emb)
        scale, shift = t_out.chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.res    = ResBlock(in_ch, out_ch, time_dim)
        self.down   = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        skip = self.res(x, t_emb)
        x    = self.down(skip)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, time_dim):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)

class UNet(nn.Module):
    def __init__(self, img_ch=3, model_dim=64, time_emb_dim=256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(img_ch, model_dim, 3, padding=1)

        self.down1 = DownBlock(model_dim,     model_dim * 2, time_emb_dim)
        self.down2 = DownBlock(model_dim * 2, model_dim * 4, time_emb_dim)
        self.down3 = DownBlock(model_dim * 4, model_dim * 4, time_emb_dim)

        self.bottleneck = nn.Sequential(
            ResBlock(model_dim * 4, model_dim * 4, time_emb_dim),
            ResBlock(model_dim * 4, model_dim * 4, time_emb_dim),
        )

        self.up3 = UpBlock(model_dim * 4, model_dim * 4, model_dim * 4, time_emb_dim)
        self.up2 = UpBlock(model_dim * 4, model_dim * 2, model_dim * 4, time_emb_dim)
        self.up1 = UpBlock(model_dim * 2, model_dim,     model_dim * 2, time_emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, model_dim),
            nn.SiLU(),
            nn.Conv2d(model_dim, img_ch, 1)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x  = self.init_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)

        for block in self.bottleneck:
            x = block(x, t_emb)

        x = self.up3(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        return self.out_conv(x)

# ──────────────────────────────────────────────────────────────────────────────
# 2. DIFFUSION SAMPLING HELPER
# ──────────────────────────────────────────────────────────────────────────────

def get_sampling_constants(timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """Precomputes DDPM noise schedule constants."""
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Required calculation variants for extraction during reverse loop
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]]) ) / (1.0 - alphas_cumprod)
    
    return betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance

@torch.no_grad()
def sample_ddpm(model, timesteps, constants, device, img_size=64, channels=3):
    """Generates a novel image starting from pure Gaussian noise."""
    betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance = constants
    
    # Start from pure noise x_T
    x = torch.randn((1, channels, img_size, img_size), device=device)
    
    # Progress bar for Streamlit UI tracking
    progress_bar = st.progress(0.0)
    
    for i in reversed(range(0, timesteps)):
        t = torch.tensor([i], device=device, dtype=torch.long)
        
        # Predict noise using the U-Net
        predicted_noise = model(x, t)
        
        # Retrieve needed schedule constants for step t
        alpha_t = sqrt_recip_alphas[i]
        beta_t = betas[i]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
        
        # Compute mean target equation
        model_mean = alpha_t * (x - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if i > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance[i]) * noise
            x = model_mean + variance
        else:
            x = model_mean
            
        # Update progress bar
        progress_bar.progress((timesteps - i) / timesteps)
        
    progress_bar.empty() # Clear when complete
    
    # Post-process image to standard uint8 format [0, 255]
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = (x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype(np.uint8)
    return Image.fromarray(x)

# ──────────────────────────────────────────────────────────────────────────────
# 3. STREAMLIT APPLICATION USER INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DDPM Face Generator", layout="centered")

st.title("DDPM Image Generation Dashboard")
st.caption("Generate realistic synthetic images using a custom PyTorch DDPM U-Net pipeline.")

# Use GPU acceleration if system infrastructure permits
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model using Cached mechanisms so it doesn't re-trigger on user clicks
@st.cache_resource
def load_trained_model():
    # Structural setup matching architecture design dimensions (img_ch=3, model_dim=64, time_dim=256)
    net = UNet(img_ch=3, model_dim=64, time_emb_dim=256)
    try:
        # Load weights mapping to current hardware infrastructure
        state_dict = torch.load("ddpm_unet.pth", map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        return net, True
    except Exception as e:
        return str(e), False

model, success = load_trained_model()

# Create layout columns for parameters and reference history visual logs
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Generation Settings")
    total_steps = st.slider("Diffusion Steps (T)", min_value=100, max_value=1000, value=1000, step=100)
    
    if not success:
        st.error(f"Failed to find or load `ddpm_unet.pth`. Error: {model}")
        generate_btn = st.button("Generate Image", disabled=True)
    else:
        st.success("Model Weights Loaded Successfully!")
        generate_btn = st.button("✨ Generate Face Image", type="primary")

with right_col:
    st.subheader("Model Metrics Logs")
    # Utilizing pre-rendered tracking visualizations present in your directory structure
    try:
        st.image("assets/loss_curve.png", caption="Training Convergence Loss Curve Tracker", use_container_width=True)
    except:
        st.info("💡 Note: Place your training `loss_curve.png` here to show historical performance.")



if generate_btn and success:
    st.subheader("" \
    "Generating Asset...")
    with st.spinner("Processing Reverse Diffusion Markov Chain steps..."):
        # Fetch configurations
        schedule_constants = get_sampling_constants(timesteps=total_steps, device=device)
        
        # Run process
        generated_img = sample_ddpm(model, timesteps=total_steps, constants=schedule_constants, device=device)
        
        # Present Output
        st.success("Generation Complete!")
        st.image(generated_img, caption=f"Generated Shape Output (Dim: 64x64 over {total_steps} Steps)", width=300)