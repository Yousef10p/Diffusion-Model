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
# 2. DIFFUSION SAMPLING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_sampling_constants(timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """Precomputes DDPM noise schedule constants."""
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Required calculation variants for extraction during reverse loop
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # Added for forward diffusion
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]]) ) / (1.0 - alphas_cumprod)
    
    return betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, sqrt_alphas_cumprod

@torch.no_grad()
def sample_ddpm(model, timesteps, constants, device, batch_size=1, img_size=64, channels=3):
    """Generates a batch of novel images starting from pure Gaussian noise."""
    betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, _ = constants
    
    # Start from pure noise for the entire batch
    x = torch.randn((batch_size, channels, img_size, img_size), device=device)
    progress_bar = st.progress(0.0)
    
    for i in reversed(range(0, timesteps)):
        # Provide time tensor mapped to the batch size explicitly
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
        predicted_noise = model(x, t)
        
        alpha_t = sqrt_recip_alphas[i]
        beta_t = betas[i]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
        
        model_mean = alpha_t * (x - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        if i > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance[i]) * noise
            x = model_mean + variance
        else:
            x = model_mean
            
        progress_bar.progress((timesteps - i) / timesteps)
        
    progress_bar.empty()
    
    x = (x.clamp(-1, 1) + 1) / 2.0
    x_np = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    
    # Return list of PIL Images
    return [Image.fromarray(img) for img in x_np]

@torch.no_grad()
def sample_ddpm_reconstruct(model, x_start, start_t, constants, device):
    """Adds noise to an image up to step start_t, then reconstructs it."""
    betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, sqrt_alphas_cumprod = constants
    
    # 1. Forward Diffusion: Add noise to the clean image up to step `start_t`
    noise = torch.randn_like(x_start)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[start_t]
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[start_t]
    
    x_noisy = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
    x = x_noisy.clone() # This is the starting point for reverse diffusion
    
    # Prepare the noisy image for Streamlit display
    display_noisy = (x_noisy.clamp(-1, 1) + 1) / 2.0
    display_noisy = (display_noisy.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype(np.uint8)
    img_noisy = Image.fromarray(display_noisy)

    # 2. Reverse Diffusion: Denoise back from `start_t` to 0
    progress_bar = st.progress(0.0)
    
    for i in reversed(range(0, start_t)):
        t = torch.tensor([i], device=device, dtype=torch.long)
        
        predicted_noise = model(x, t)
        
        alpha_t = sqrt_recip_alphas[i]
        beta_t = betas[i]
        sqrt_one_minus_alpha_cumprod_i = sqrt_one_minus_alphas_cumprod[i]
        
        model_mean = alpha_t * (x - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_i)
        
        if i > 0:
            step_noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance[i]) * step_noise
            x = model_mean + variance
        else:
            x = model_mean
            
        progress_bar.progress((start_t - i) / start_t)
        
    progress_bar.empty()
    
    x = (x.clamp(-1, 1) + 1) / 2.0
    x = (x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype(np.uint8)
    return img_noisy, Image.fromarray(x)

# ──────────────────────────────────────────────────────────────────────────────
# 3. STREAMLIT APPLICATION USER INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DDPM Generator", layout="centered")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_trained_model():
    net = UNet(img_ch=3, model_dim=64, time_emb_dim=256)
    try:
        state_dict = torch.load("ddpm_unet.pth", map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        return net, True
    except Exception as e:
        return str(e), False

model, success = load_trained_model()

# ── Navigation Sidebar ──
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode:", ["Unconditional Generation"])

if app_mode == "Unconditional Generation":
    st.title("DDPM Image Generation Dashboard")
    st.caption("Generate realistic synthetic images using a custom PyTorch DDPM U-Net pipeline.")

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Generation Settings")
        total_steps = st.slider("Diffusion Steps (T)", min_value=100, max_value=1000, value=1000, step=100)
        batch_size = st.number_input("Number of Images (Batch Size)", min_value=1, max_value=16, value=1)
        
        if not success:
            st.error(f"Failed to find or load `ddpm_unet.pth`. Error: {model}")
            generate_btn = st.button("Generate Image(s)", disabled=True)
        else:
            st.success("Model Weights Loaded Successfully!")
            generate_btn = st.button("✨ Generate Image(s)", type="primary")

    with right_col:
        st.subheader("Model Metrics Logs")
        try:
            st.image("assets/loss_curve.png", caption="Training Convergence Loss Curve Tracker", use_container_width=True)
        except:
            st.info("💡 Note: Place your training `loss_curve.png` here to show historical performance.")

    if generate_btn and success:
        st.subheader("Generating Assets...")
        with st.spinner(f"Processing Reverse Diffusion over {batch_size} image(s)..."):
            schedule_constants = get_sampling_constants(timesteps=total_steps, device=device)
            generated_images = sample_ddpm(
                model, 
                timesteps=total_steps, 
                constants=schedule_constants, 
                device=device,
                batch_size=batch_size
            )
            
            st.success("Generation Complete!")
            
            # Display generated batch cleanly in columns (up to 4 images per row)
            cols = st.columns(min(batch_size, 4))
            for idx, img in enumerate(generated_images):
                with cols[idx % 4]:
                    st.image(img, use_container_width=True, caption=f"Sample {idx+1}")

elif app_mode == "Image Reconstruction":
    st.title("DDPM Image Reconstruction")
    st.caption("Upload an image, apply forward diffusion (noise), and use the U-Net to denoise and reconstruct it.")

    if not success:
        st.error(f"Failed to find or load `ddpm_unet.pth`. Error: {model}")
    else:
        uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
        noise_timestep = st.slider("Noise Level (Timestep)", min_value=10, max_value=999, value=300, step=10)
        
        if uploaded_file is not None:
            # Preprocess the uploaded image
            raw_img = Image.open(uploaded_file).convert("RGB")
            resized_img = raw_img.resize((64, 64))
            
            # Show original resized image
            st.subheader("Original Image (Resized to 64x64)")
            st.image(resized_img, width=200)

            # Convert to normalized tensor [-1, 1]
            img_np = np.array(resized_img) / 255.0
            img_np = img_np * 2.0 - 1.0
            x_start = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            reconstruct_btn = st.button("🔄 Add Noise & Reconstruct", type="primary")

            if reconstruct_btn:
                with st.spinner("Applying Forward & Reverse Diffusion..."):
                    # We pass 1000 for total sequence to get correct schedules, but start diffusing from `noise_timestep`
                    schedule_constants = get_sampling_constants(timesteps=1000, device=device)
                    
                    noisy_img, reconstructed_img = sample_ddpm_reconstruct(
                        model, 
                        x_start, 
                        start_t=noise_timestep, 
                        constants=schedule_constants, 
                        device=device
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"Noised (t={noise_timestep})")
                        st.image(noisy_img, use_container_width=True)
                    with col2:
                        st.subheader("Reconstructed")
                        st.image(reconstructed_img, use_container_width=True)