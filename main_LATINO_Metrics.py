"""
Batch evaluation variant of main_LATINO.py.

Runs LATINO (or other solvers) over every image in a dataset directory,
computes PSNR / SSIM / LPIPS for each image, and writes a single
summary CSV.  No per-image plots or sample images are saved — only:

  <output_dir>/
      results.csv          – per-image metrics  (image, PSNR, SSIM, LPIPS)
      summary.csv          – dataset-level mean ± std

Usage:
    python main_LATINO_Metric.py image.path=<dir_of_images> [image.prompt="a photo of a face"]

The `image.path` override should point to a *directory* of images
(png/jpg/jpeg).  Everything else is read from configs/LATINO.yaml as
usual.
"""

import os
import glob
import json
import time
import csv
import random

import deepinv as dinv
import torch
import torchmetrics
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra

from transformers import CLIPProcessor, CLIPModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline,
    LCMScheduler,
    AutoPipelineForText2Image,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from huggingface_hub import hf_hub_download

from utils import (
    load_image_tensor,
    crop_to_multiple,
    get_filename_from_path,
    _get_x_init,
)
from inverse_problems import get_forward_model
from noise_schemes import (
    noise_pred_cond_y,
    noise_pred_cond_y_15,
    noise_pred_cond_y_DPS,
    noise_pred_cond_y_PSLD,
    noise_pred_cond_y_DPS_P2L,
    noise_pred_cond_y_DPS_1024,
    noise_pred_cond_y_PSLD_1024,
    noise_pred_cond_y_DPS_1024_P2L,
    noise_pred_cond_y_TReg,
)


def collect_image_paths(directory):
    """Return sorted list of image file paths from a directory."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    paths.sort()
    return paths


def run_single_image(
    cfg,
    image_path,
    pipe,
    device,
    text_embeddings,
    uncond_embeddings,
    # SDXL-specific (None for v1.5 models)
    added_cond_kwargs=None,
    uncond_added_cond_kwargs=None,
    custom_timesteps=None,
    image_height=None,
    image_width=None,
    # v1.5-specific (None for SDXL models)
    text_embeddings_cfg=None,
    guidance_scale=None,
    height=None,
    width=None,
    num_inference_steps=None,
    # CLIP / TReg
    clip_model=None,
    clip_processor=None,
    openclip_model=None,
    openclip_preprocess=None,
):
    """Run the solver on a single image and return (psnr, ssim, lpips)."""

    seed = cfg.seed

    # ── Load and prepare clean image ──────────────────────────────
    xtemp = load_image_tensor(image_path)
    x_clean = crop_to_multiple(xtemp, m=8).to(device)

    if cfg.model not in {"LATINO", "LDPS1024", "PSLD1024", "LDPS1024-P2L", "TREG1024"}:
        if xtemp.shape[-1] == 1024:
            noise_model = dinv.physics.GaussianNoise(sigma=0)
            model_down = dinv.physics.Downsampling(
                img_size=(3, 1024, 1024), factor=2, device=device,
                noise_model=noise_model, filter="bicubic",
            )
            x_clean = model_down(x_clean).clamp(0, 1)
    else:
        if xtemp.shape[-1] == 512:
            noise_model = dinv.physics.GaussianNoise(sigma=0)
            model_up = dinv.physics.Downsampling(
                img_size=(3, 1024, 1024), factor=2, device=device,
                noise_model=noise_model, filter="bicubic",
            ).A_adjoint
            x_clean = model_up(x_clean).clamp(0, 1)

    x_clean = (x_clean - x_clean.min()) / (x_clean.max() - x_clean.min())

    # ── Forward model ─────────────────────────────────────────────
    forward_model, transpose_operator = get_forward_model(cfg, x_clean, device)
    y = forward_model(x_clean)
    y_norm = y * 2 - 1
    sigma_y_norm = cfg.problem.sigma_y * 2

    # ── Mask (inpainting only) ────────────────────────────────────
    mask = forward_model.mask if cfg.problem.type == "inpainting_squared_mask" else None

    # ── Initialisation ────────────────────────────────────────────
    mu_z = None
    latents_base = None

    if cfg.init_strategy == "y_noise":
        x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
        with torch.no_grad():
            qz = (
                pipe.vae.encode(x_init.clip(-1, 1).half())
                if cfg.model not in {"LDPS1024-P2L", "LDPS1024", "PSLD1024"}
                else pipe.vae.encode(x_init.clip(-1, 1))
            )
        mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor
    elif cfg.init_strategy == "y":
        x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
        with torch.no_grad():
            qz = (
                pipe.vae.encode(x_init.clip(-1, 1).half())
                if cfg.model not in {"LDPS1024-P2L", "LDPS1024", "PSLD1024"}
                else pipe.vae.encode(x_init.clip(-1, 1))
            )
        latents_base = qz.latent_dist.mean * pipe.vae.config.scaling_factor

    # P2L Adam hyperparameters
    if cfg.model in {"LDPS-P2L", "LDPS1024-P2L"}:
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        lr_prompt = 1e-4
        lr_grad = 0.05

    num_samples = cfg.get("num_samples", 1)
    all_samples = []

    lpips_fn = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity("vgg").to(device)
    psnr_fn = torchmetrics.image.PeakSignalNoiseRatio(data_range=1).to(device)
    ssim_fn = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).to(device)

    # Need a fresh copy of text embeddings per image for models that modify them
    te = text_embeddings.clone()
    ue = uncond_embeddings.clone()

    for sample_idx in range(num_samples):
        sample_seed = seed + sample_idx

        # ── Re-initialise latents ─────────────────────────────────
        if cfg.init_strategy == "y_noise":
            gen = torch.Generator(device=device).manual_seed(sample_seed)
            noise_s = torch.randn(mu_z.shape, generator=gen, device=device, dtype=mu_z.dtype)
            latents = pipe.scheduler.add_noise(mu_z, noise=noise_s, timesteps=torch.tensor([999]))
        elif cfg.init_strategy == "y":
            latents = latents_base.clone()
        else:
            gen = torch.Generator(device=device).manual_seed(sample_seed)
            if cfg.model in {"LATINO", "LDPS1024", "PSLD1024", "LDPS1024-P2L", "TREG1024"}:
                latents = pipe.prepare_latents(
                    batch_size=1,
                    num_channels_latents=pipe.unet.config.in_channels,
                    height=image_height, width=image_width,
                    dtype=torch.float16 if cfg.model == "LATINO" else torch.float32,
                    device=device, generator=gen,
                )
            else:
                latents = torch.randn(
                    (1, pipe.unet.config.in_channels, height // 8, width // 8),
                    device=device, dtype=torch.float16, generator=gen,
                )

        # Reset P2L Adam state
        if cfg.model in {"LDPS-P2L", "LDPS1024-P2L"}:
            m = torch.zeros_like(te, dtype=torch.float32)
            v = torch.zeros_like(te, dtype=torch.float32)
            t_step = 0
            m2 = torch.zeros_like(latents, dtype=torch.float32)
            v2 = torch.zeros_like(latents, dtype=torch.float32)
            t_step2 = 0

        # Reset scheduler
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        if cfg.model == "LATINO":
            pipe.scheduler.timesteps = custom_timesteps

        # ── Diffusion loop ────────────────────────────────────────
        for i, timestep in enumerate(pipe.scheduler.timesteps):
            if cfg.model == "LATINO":
                te_step = te.detach().requires_grad_(True)
                with torch.no_grad():
                    noise_uncond = pipe.unet(
                        latents, timestep,
                        encoder_hidden_states=te_step,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                with torch.no_grad():
                    _, noise_pred = noise_pred_cond_y(
                        latents=latents, t=timestep, pipe=pipe, cfg=cfg,
                        logdir="", y_guidance=y_norm,
                        forward_model=forward_model, noise_pred=noise_uncond,
                        sigma_y=sigma_y_norm,
                    )

            elif cfg.model == "LATINO-1.5":
                te_cfg = torch.cat([ue, te], dim=0)
                with torch.no_grad():
                    noise_pred = noise_pred_cond_y_15(
                        latents=latents, t=timestep,
                        encoder_hidden_states=te_cfg,
                        guidance_scale=guidance_scale, pipe=pipe, cfg=cfg,
                        logdir="", y_guidance=y_norm,
                        forward_model=forward_model, sigma_y=sigma_y_norm,
                    )

            elif cfg.model == "LDPS":
                te_cfg = torch.cat([ue, te], dim=0)
                noise_pred, grad_nll = noise_pred_cond_y_DPS(
                    latents=latents, t=timestep,
                    encoder_hidden_states=te_cfg,
                    guidance_scale=guidance_scale, pipe=pipe,
                    logdir="", y_guidance=y_norm,
                    forward_model=forward_model,
                )

            elif cfg.model == "LDPS1024":
                noise_pred, grad_nll = noise_pred_cond_y_DPS_1024(
                    latents=latents, t=timestep, pipe=pipe,
                    text_embeddings=te,
                    added_cond_kwargs=added_cond_kwargs,
                    logdir="", y_guidance=y_norm,
                    forward_model=forward_model,
                )

            elif cfg.model == "PSLD":
                te_cfg = torch.cat([ue, te], dim=0)
                noise_pred, grad_nll = noise_pred_cond_y_PSLD(
                    latents=latents, t=timestep,
                    encoder_hidden_states=te_cfg,
                    guidance_scale=guidance_scale, pipe=pipe,
                    logdir="", y_guidance=y_norm,
                    forward_model=forward_model,
                    transpose_model=transpose_operator,
                )

            elif cfg.model == "PSLD1024":
                noise_pred, grad_nll = noise_pred_cond_y_PSLD_1024(
                    latents=latents, t=timestep, pipe=pipe,
                    text_embeddings=te,
                    added_cond_kwargs=added_cond_kwargs,
                    logdir="", y_guidance=y_norm,
                    forward_model=forward_model,
                    transpose_model=transpose_operator,
                )

            elif cfg.model == "LDPS-P2L":
                te_cfg_local = torch.cat([ue, te], dim=0)
                for _ in range(1):
                    te = te.detach().requires_grad_(True)
                    te_cfg_local = torch.cat([ue, te], dim=0)
                    with torch.enable_grad():
                        latent_model_input = torch.cat([latents] * 2, dim=0)
                        t_tensor = torch.tensor([timestep], dtype=torch.float16).to(device)
                        np_out = pipe.unet(latent_model_input, t_tensor, encoder_hidden_states=te_cfg_local).sample
                        noise_pred_uncond, noise_pred_text = np_out.chunk(2)
                        noise_unc = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        alpha_t = pipe.scheduler.alphas_cumprod[timestep]
                        z0 = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_unc)
                        x_dec = pipe.vae.decode(z0 / pipe.vae.config.scaling_factor).sample.clip(-1, 1)
                        x_dec = (x_dec + 1) / 2
                        model_output = forward_model(x_dec.float()).clamp(-1, 1)
                        loss = torch.norm(y_norm - model_output)
                        gradients = torch.autograd.grad(loss, inputs=te)[0]
                    t_step += 1
                    m = beta1 * m + (1 - beta1) * gradients
                    v = beta2 * v + (1 - beta2) * (gradients ** 2)
                    m_hat = m / (1 - beta1 ** t_step)
                    v_hat = v / (1 - beta2 ** t_step)
                    v_hat = torch.clamp(v_hat, min=epsilon)
                    te = te - (lr_prompt / torch.sqrt(v_hat)) * m_hat
                    te = te.to(torch.float16)

                noise_pred, grad_nll = noise_pred_cond_y_DPS_P2L(
                    latents=latents, t=timestep,
                    encoder_hidden_states=te_cfg_local,
                    guidance_scale=guidance_scale, pipe=pipe,
                    logdir="", y_guidance=y_norm,
                    forward_model=forward_model,
                )

            elif cfg.model == "LDPS1024-P2L":
                with torch.autocast(device_type="cuda", enabled=False):
                    for _ in range(5):
                        te = te.detach().requires_grad_(True)
                        with torch.enable_grad():
                            noise_unc = pipe.unet(
                                latents, timestep,
                                encoder_hidden_states=te.float(),
                                added_cond_kwargs=added_cond_kwargs,
                            ).sample
                            alpha_t = pipe.scheduler.alphas_cumprod[timestep]
                            z0 = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_unc)
                            x_dec = pipe.vae.decode(z0 / pipe.vae.config.scaling_factor).sample.clip(-1, 1)
                            x_dec = (x_dec + 1) / 2
                            model_output = forward_model(x_dec.float()).clamp(-1, 1)
                            loss = torch.norm(y_norm - model_output)
                            gradients = torch.autograd.grad(loss, inputs=te, retain_graph=False)[0]
                        t_step += 1
                        m = beta1 * m + (1 - beta1) * gradients
                        v = beta2 * v + (1 - beta2) * (gradients ** 2)
                        m_hat = m / (1 - beta1 ** t_step)
                        v_hat = v / (1 - beta2 ** t_step)
                        v_hat = torch.clamp(v_hat, min=epsilon)
                        te = te - (lr_prompt / torch.sqrt(v_hat)) * m_hat

                noise_pred, grad_nll = noise_pred_cond_y_DPS_1024_P2L(
                    latents=latents, t=timestep, pipe=pipe,
                    text_embeddings=te,
                    added_cond_kwargs=added_cond_kwargs,
                    logdir="", y_guidance=y_norm,
                    forward_model=forward_model,
                )

            elif cfg.model == "TREG":
                with torch.no_grad():
                    skip = 5
                    prev_t = timestep - skip
                    alpha_t = pipe.scheduler.alphas_cumprod[timestep]
                    at_prev = pipe.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else pipe.scheduler.final_alpha_cumprod.to(device)
                    latent_model_input = torch.cat([latents] * 2, dim=0)
                    t_tensor = torch.tensor(timestep, dtype=torch.float16).to(device)
                    t_in = torch.cat([t_tensor.unsqueeze(0)] * 2)
                    te_cfg = torch.cat([ue.half(), te], dim=0)
                    noise_pred = pipe.unet(latent_model_input, t_in, encoder_hidden_states=te_cfg).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if timestep % 3 == 0 and timestep < 850:
                    with torch.enable_grad():
                        with torch.no_grad():
                            z0_pred_c = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                            x_dec = pipe.vae.decode(z0_pred_c / pipe.vae.config.scaling_factor).sample.clip(-1, 1)
                            z0_predy, x_dec = noise_pred_cond_y_TReg(
                                x=x_dec, z0_pred=z0_pred_c, pipe=pipe,
                                y_guidance=y_norm, forward_model=forward_model,
                            )
                            x_clip = ((x_dec + 1) / 2).clip(0, 1)
                        image_tensor = clip_processor(images=x_clip.float(), return_tensors="pt").to(device)["pixel_values"]
                        img_feats = clip_model.get_image_features(pixel_values=image_tensor)
                        img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
                        lr_t = 1e-3
                        ue = ue.to(torch.float32).clone().detach().requires_grad_(True)
                        optim_text = torch.optim.Adam([ue], lr=lr_t)
                        for _ in range(10):
                            optim_text.zero_grad()
                            sim = img_feats @ ue.permute(0, 2, 1)
                            loss = sim.mean()
                            loss.backward(retain_graph=True)
                            optim_text.step()
                        noise = torch.randn_like(z0_predy).to(device)
                        z0_ema = at_prev * z0_predy + (1 - at_prev) * z0_pred_c
                        latents = at_prev.sqrt() * z0_ema + (1 - at_prev) * noise_pred
                        latents = latents + (1 - at_prev).sqrt() * at_prev.sqrt() * noise
                else:
                    with torch.no_grad():
                        z0t = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                        latents = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred

            elif cfg.model == "TREG1024":
                with torch.no_grad():
                    skip = 5
                    prev_t = timestep - skip
                    alpha_t = pipe.scheduler.alphas_cumprod[timestep]
                    at_prev = pipe.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else pipe.scheduler.final_alpha_cumprod.to(device)
                with torch.no_grad():
                    noise_pred = pipe.unet(
                        latents, timestep, encoder_hidden_states=te,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    noise_uncond = pipe.unet(
                        latents, timestep, encoder_hidden_states=ue.half(),
                        added_cond_kwargs=uncond_added_cond_kwargs,
                    ).sample
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                if timestep % 3 == 0 and timestep < 850:
                    with torch.enable_grad():
                        with torch.no_grad():
                            z0_pred_c = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                            x_dec = pipe.vae.decode(z0_pred_c / pipe.vae.config.scaling_factor).sample.clip(-1, 1)
                            z0_predy, x_dec = noise_pred_cond_y_TReg(
                                x=x_dec, z0_pred=z0_pred_c, pipe=pipe,
                                y_guidance=y_norm, forward_model=forward_model,
                            )
                            x_clip = ((x_dec + 1) / 2).clip(0, 1)
                        from torchvision.transforms.functional import to_pil_image
                        image_tensor = clip_processor(images=x_clip.float(), return_tensors="pt").to(device)["pixel_values"]
                        clip_features = clip_model.get_image_features(pixel_values=image_tensor)
                        clip_features = clip_features / clip_features.norm(dim=1, keepdim=True)
                        pil_image = to_pil_image(x_clip[0].cpu())
                        image_tensor2 = openclip_preprocess(pil_image).unsqueeze(0).to(device)
                        openclip_features = openclip_model.encode_image(image_tensor2)
                        openclip_features = openclip_features / openclip_features.norm(dim=-1, keepdim=True)
                        combined_features = torch.cat((openclip_features, clip_features), dim=-1)
                        lr_t = 1e-3
                        ue = ue.to(torch.float32).clone().detach().requires_grad_(True)
                        optim_text = torch.optim.Adam([ue], lr=lr_t)
                        for _ in range(10):
                            optim_text.zero_grad()
                            sim = combined_features @ ue.permute(0, 2, 1)
                            loss = sim.mean()
                            loss.backward(retain_graph=True)
                            optim_text.step()
                        noise = torch.randn_like(z0_predy).to(device)
                        z0_ema = at_prev * z0_predy + (1 - at_prev) * z0_pred_c
                        latents = at_prev.sqrt() * z0_ema + (1 - at_prev) * noise_pred
                        latents = latents + (1 - at_prev).sqrt() * at_prev.sqrt() * noise
                else:
                    with torch.no_grad():
                        z0t = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                        latents = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred

            # Step the scheduler
            if cfg.model not in {"TREG", "TREG1024"}:
                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

            if cfg.model in {"LDPS", "PSLD", "LDPS1024", "PSLD1024"}:
                latents -= grad_nll
            elif cfg.model in {"LDPS1024-P2L", "LDPS-P2L"}:
                if cfg.problem.type == "super_resolution_bicubic":
                    latents -= lr_grad * grad_nll
                else:
                    t_step2 += 1
                    m2 = beta1 * m2 + (1 - beta1) * grad_nll
                    v2 = beta2 * v2 + (1 - beta2) * (grad_nll ** 2)
                    m_hat2 = m2 / (1 - beta1 ** t_step)
                    v_hat2 = v2 / (1 - beta2 ** t_step)
                    v_hat2 = torch.clamp(v_hat2, min=epsilon)
                    latents = latents - (lr_grad / (torch.sqrt(v_hat2) + epsilon)) * m_hat2
                    if cfg.model == "LDPS-P2L":
                        latents = latents.to(torch.float16)

        # Decode sample
        with torch.no_grad():
            decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        restored_x = (decoded / 2 + 0.5).clamp(0, 1)
        all_samples.append(restored_x.detach().cpu())

    # ── Compute metrics on posterior mean ─────────────────────────
    all_samples_t = torch.cat(all_samples, dim=0)
    posterior_mean = all_samples_t.mean(0, keepdim=True).to(device)

    psnr = psnr_fn(posterior_mean, x_clean).item()
    ssim = ssim_fn(posterior_mean, x_clean).item()
    lpips = lpips_fn(posterior_mean * 2 - 1, x_clean * 2 - 1).item()

    # Free GPU memory
    del all_samples, all_samples_t, posterior_mean, x_clean
    torch.cuda.empty_cache()

    return psnr, ssim, lpips


@hydra.main(version_base=None, config_path="configs", config_name="LATINO")
def main(cfg: DictConfig) -> None:
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    # ── Resolve dataset directory ─────────────────────────────────
    dataset_dir = cfg.image.path
    if not os.path.isdir(dataset_dir):
        raise ValueError(
            f"image.path must point to a directory of images for batch evaluation. "
            f"Got: {dataset_dir}"
        )

    image_paths = collect_image_paths(dataset_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_dir}")
    print(f"Found {len(image_paths)} images in {dataset_dir}")

    # ── Load CLIP (TReg only) ─────────────────────────────────────
    clip_model = clip_processor = openclip_model = openclip_preprocess = None
    if cfg.model in {"TREG", "TREG1024"}:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if cfg.model == "TREG1024":
            import open_clip
            openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms(
                "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device,
            )

    # ── Load pipeline (identical to main_LATINO.py) ───────────────
    custom_timesteps = None
    added_cond_kwargs = None
    uncond_added_cond_kwargs = None
    image_height = image_width = None
    text_embeddings_cfg = None
    guidance_scale = None
    height = width = None
    num_inference_steps = None

    if cfg.model in {"LATINO", "LDPS1024", "PSLD1024", "LDPS1024-P2L", "TREG1024"}:
        base_model_id = "/lustre/fswork/projects/rech/ynx/uxl64xr/.cache/huggingface/models/sdxl"

        if cfg.model == "LATINO":
            repo_name = "tianweiy/DMD2"
            ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
            unet_config = UNet2DConditionModel.load_config(base_model_id, subfolder="unet")
            unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
            unet.load_state_dict(torch.load("/lustre/fswork/projects/rech/ynx/uxl64xr/.cache/huggingface/hub/models--tianweiy--DMD2/snapshots/be22767697a1f3ca656b73c776e15fa335c86c6c/dmd2_sdxl_4step_unet_fp16.bin", map_location=device, weights_only=True))
            vae = AutoencoderKL.from_pretrained("/lustre/fswork/projects/rech/ynx/uxl64xr/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/207b116dae70ace3637169f1ddd2434b91b3a8cd", torch_dtype=torch.float16)
            pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, vae=vae, torch_dtype=torch.float16, variant="fp16", guidance_scale=0).to(device)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        elif cfg.model == "TREG1024":
            base_model_id = "/lustre/fswork/projects/rech/ynx/uxl64xr/.cache/huggingface/models/sdxl"
            vae = AutoencoderKL.from_pretrained("/lustre/fswork/projects/rech/ynx/uxl64xr/.cache/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/207b116dae70ace3637169f1ddd2434b91b3a8cd", torch_dtype=torch.float16)
            pipe = DiffusionPipeline.from_pretrained(base_model_id, vae=vae, torch_dtype=torch.float16, use_safetensors=True).to(device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        else:
            base_model_id = "/lustre/fswork/projects/rech/ynx/uxl64xr/.cache/huggingface/models/sdxl"
            pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32, guidance_scale=0).to(device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        prompt = cfg.image.prompt
        text_embeddings, _, pooled_text_embeds, _ = pipe.encode_prompt(prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        uncond_embeddings, _, uncond_pooled_text_embeds, _ = pipe.encode_prompt(
            "" if cfg.model != "TREG1024" else "out of focus, depth of field",
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
        )

        generator = torch.Generator(device=device).manual_seed(seed)
        image_height, image_width = 1024, 1024

        time_ids = pipe._get_add_time_ids(
            original_size=(image_height, image_width),
            crops_coords_top_left=(0, 0),
            target_size=(image_height, image_width),
            dtype=torch.float16 if cfg.model == "LATINO" else torch.float32,
            text_encoder_projection_dim=1280,
        ).to(device)

        added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": time_ids}
        uncond_added_cond_kwargs = {"text_embeds": uncond_pooled_text_embeds, "time_ids": time_ids}

        if cfg.model == "LATINO":
            num_inference_steps = 8
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
            custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
            pipe.scheduler.timesteps = custom_timesteps
        elif cfg.model == "TREG1024":
            num_inference_steps = 200
            guidance_scale = 5.0
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        else:
            num_inference_steps = 500
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    else:
        if cfg.model != "LATINO-1.5":
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            if cfg.model == "TREG":
                num_inference_steps = 200
                guidance_scale = 5.0
            else:
                num_inference_steps = 999
                guidance_scale = 1
        else:
            model_id = "runwayml/stable-diffusion-v1-5"
            adapter_id = "latent-consistency/lcm-lora-sdv1-5"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.to("cuda")
            pipe.load_lora_weights(adapter_id)
            pipe.fuse_lora()
            num_inference_steps = 8
            guidance_scale = 1

        unet = pipe.unet
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        prompt = [cfg.image.prompt]
        text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

        if cfg.model == "TREG":
            uncond_inputs = tokenizer(
                ["out of focus, depth of field"], padding="max_length",
                max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",
            )
        else:
            uncond_inputs = tokenizer(
                [""] * len(prompt), padding="max_length",
                max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",
            )
        uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]

        height, width = 512, 512
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        text_embeddings_cfg = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    # ── Output directory ──────────────────────────────────────────
    output_dir = os.path.join(os.curdir, f"batch_results_{cfg.model}_{cfg.problem.type}")
    os.makedirs(output_dir, exist_ok=True)

    results_csv = os.path.join(output_dir, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "PSNR", "SSIM", "LPIPS"])

    # ── Main loop over images ─────────────────────────────────────
    all_psnr, all_ssim, all_lpips = [], [], []
    start_time = time.time()

    for idx, img_path in enumerate(image_paths):
        imname = get_filename_from_path(img_path)
        print(f"\n[{idx + 1}/{len(image_paths)}] Processing {imname} ...")
        t0 = time.time()

        psnr, ssim, lpips = run_single_image(
            cfg=cfg,
            image_path=img_path,
            pipe=pipe,
            device=device,
            text_embeddings=text_embeddings,
            uncond_embeddings=uncond_embeddings,
            added_cond_kwargs=added_cond_kwargs,
            uncond_added_cond_kwargs=uncond_added_cond_kwargs,
            custom_timesteps=custom_timesteps,
            image_height=image_height,
            image_width=image_width,
            text_embeddings_cfg=text_embeddings_cfg,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            clip_model=clip_model,
            clip_processor=clip_processor,
            openclip_model=openclip_model,
            openclip_preprocess=openclip_preprocess,
        )

        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(lpips)

        print(f"  {imname} — PSNR: {psnr:.3f}  SSIM: {ssim:.4f}  LPIPS: {lpips:.4f}  ({time.time() - t0:.1f}s)")

        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([imname, psnr, ssim, lpips])

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────
    summary = {
        "num_images": len(image_paths),
        "num_samples_per_image": cfg.get("num_samples", 1),
        "model": cfg.model,
        "problem": cfg.problem.type,
        "PSNR_mean": float(np.mean(all_psnr)),
        "PSNR_std": float(np.std(all_psnr)),
        "SSIM_mean": float(np.mean(all_ssim)),
        "SSIM_std": float(np.std(all_ssim)),
        "LPIPS_mean": float(np.mean(all_lpips)),
        "LPIPS_std": float(np.std(all_lpips)),
        "total_time_s": round(elapsed, 1),
    }

    summary_csv = os.path.join(output_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\n" + "=" * 60)
    print(f"  Dataset evaluation complete  ({len(image_paths)} images, {elapsed:.0f}s)")
    print(f"  Model: {cfg.model}  |  Problem: {cfg.problem.type}")
    print(f"  PSNR:  {summary['PSNR_mean']:.3f} ± {summary['PSNR_std']:.3f}")
    print(f"  SSIM:  {summary['SSIM_mean']:.4f} ± {summary['SSIM_std']:.4f}")
    print(f"  LPIPS: {summary['LPIPS_mean']:.4f} ± {summary['LPIPS_std']:.4f}")
    print(f"  Results: {results_csv}")
    print(f"  Summary: {summary_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
