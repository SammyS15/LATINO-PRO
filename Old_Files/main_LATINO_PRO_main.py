import os
import json
import time
import deepinv as dinv
from torchvision.utils import save_image
import random
import torch
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from diffusers import (
    AutoPipelineForText2Image,
    LCMScheduler,
)
from omegaconf import DictConfig, OmegaConf
import hydra

from utils import (
    load_image_tensor,
    crop_to_multiple,
    get_filename_from_path,
    find_available_filename,
    _get_x_init,
)
from inverse_problems import get_forward_model
from noise_schemes import (
    noise_pred_cond_y_PRO,
)

@hydra.main(version_base=None, config_path="configs", config_name="LATINO-PRO")
def main(cfg: DictConfig) -> None:
    # Set global random seeds for full reproducibility
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior for CUDA (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the device
    device = torch.device("cuda")  # Use torch.device instead of string

    lpips_loss = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity('vgg').to(device)
    psnr_loss = torchmetrics.image.PeakSignalNoiseRatio(data_range=1).to(device)
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1).to(device)

    # load stable diffusion
    model_id = "/home/sammys15/links/scratch/Latent_Posterior_Sampling_Method_Comparsion/stable_diffusion_1_5_model"
    adapter_id = "/home/sammys15/links/scratch/Latent_Posterior_Sampling_Method_Comparsion/lcm-lora-sdv1-5"

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16",
                                                     requires_safety_checker=False, safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()
    pipe = pipe.to(device)

    prompt = cfg.image.prompt

    # Encode text to conditioning
    text_embeddings, _ = pipe.encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )

    # Create a random generator
    generator = torch.Generator(device=device).manual_seed(seed)

    # Assuming desired resolution of 512x512
    image_height = 512
    image_width = 512

    # Prepare initial noise latents with correct device type
    latents = pipe.prepare_latents(
        batch_size=1,  # number of images to generate
        num_channels_latents=pipe.unet.config.in_channels,  # latent channels
        height=image_height,  # image height
        width=image_width,   # image width
        dtype=torch.float16,  # datatype
        device=device,  # Corrected device type
        generator=generator  # Random number generator
    )

    # Define the number of inference steps and set timesteps
    num_inference_steps = 4     # set to 8 to use the full LATINO schedule
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Override the scheduler's timesteps with DMD2 values (or whole 8-steps)
    #custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
    custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
    pipe.scheduler.timesteps = custom_timesteps

    # load clean image
    xtemp = load_image_tensor(cfg.image.path)

    x_clean = crop_to_multiple(xtemp, m=8).to(device)
    
    # Downscale 1024x1024 inputs to 512x512 for SD1.5
    if xtemp.shape[-1] == 1024:
        noise_model_1024_to_512 = dinv.physics.GaussianNoise(sigma=0)

        model_1024_to_512 = dinv.physics.Downsampling(
                img_size=(3, 1024, 1024),
                factor=2,
                device=device,
                noise_model=noise_model_1024_to_512,
                filter="bicubic"
                )

        x_clean = model_1024_to_512(x_clean).clamp(0, 1)

    x_clean = (x_clean - x_clean.min())/(x_clean.max() - x_clean.min())

    # Build forward and transpose operators
    forward_model, transpose_operator = get_forward_model(cfg, x_clean, device)
    
    y = forward_model(x_clean)
    y_norm = y * 2 - 1
    sigma_y_norm = cfg.problem.sigma_y * 2

    # create log folder
    logdir = os.curdir

    xp_log_dir = os.path.join(logdir, "results_LATINO_PRO", cfg.problem.type, cfg.log_subfolder)

    os.makedirs(xp_log_dir, exist_ok=True)
    imname = get_filename_from_path(cfg.image.path)
    xpname = find_available_filename(folder=xp_log_dir, prefix=f'{imname}')
    xp_log_dir = os.path.join(xp_log_dir, xpname)
    print(f'logging results in {xp_log_dir}')
    os.makedirs(xp_log_dir, exist_ok=True)
    with open(os.path.join(xp_log_dir, 'config.yaml'), 'w+') as f:
        OmegaConf.save(config=cfg, f=f)

    # Apply the initialization strategy
    mu_z = None  # populated for y_noise init; kept for posterior sampling loop
    if cfg.problem.type != 'inpainting_squared_mask':
        mask = None
    else:
        mask = forward_model.mask
    if cfg.init_strategy == 'y_noise':
        x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
        save_image(x_init*0.5 + 0.5, os.path.join(xp_log_dir, 'x_init.png'))
        with torch.no_grad():
            qz = pipe.vae.encode(x_init.clip(-1, 1).half())
        mu_z = qz.latent_dist.mean * pipe.vae.config.scaling_factor
        noise = torch.randn_like(mu_z)
        latents = pipe.scheduler.add_noise(mu_z, noise=noise, timesteps=torch.tensor([999]))

    elif cfg.init_strategy == 'y':
        x_init, y_norm = _get_x_init(y_norm, forward_model, transpose_operator, mask, cfg)
        save_image(x_init*0.5 + 0.5, os.path.join(xp_log_dir, 'x_init.png'))
        with torch.no_grad():
            qz = pipe.vae.encode(x_init.clip(-1, 1).half())
        latents = qz.latent_dist.mean * pipe.vae.config.scaling_factor

    text_embeddings_0 = text_embeddings

    accumulated_grad = torch.zeros_like(text_embeddings)

    csv_file = os.path.join(xp_log_dir, "metrics_log.csv")
    write_header = True  # Change to False if the file already exists and you don't want to rewrite the header.
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if it's the first time
        if write_header:
            writer.writerow(['Iteration', 'PSNR', 'SSIM', 'LPIPS'])

    start_time = time.time()
    for j in range(cfg.num_SAPG_steps):
        print(f"SAPG step: {j + 1}")
        # Inspecting the pipeline timesteps
        for i, timestep in enumerate(pipe.scheduler.timesteps):
            print(f"Step {i + 1}: Timestep {timestep}")
            text_embeddings = text_embeddings.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                noise_uncond = pipe.unet(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample

            with torch.no_grad():
                _, noise_pred = noise_pred_cond_y_PRO(
                    latents=latents,
                    t = timestep,
                    pipe=pipe,
                    cfg=cfg,
                    logdir=xp_log_dir,
                    y_guidance=y_norm,
                    forward_model=forward_model,
                    noise_pred=noise_uncond,
                    sigma_y = sigma_y_norm,
                    SAPG_j = j,
                    n_steps = num_inference_steps
                )
            if i<num_inference_steps-1:
                alpha_t = pipe.scheduler.alphas_cumprod[timestep]
                with torch.enable_grad():
                    z0_pred_c = torch.sqrt(1 / alpha_t) * (latents - torch.sqrt(1 - alpha_t) * noise_uncond)

                with torch.no_grad():
                    latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

                def project_onto_ball(theta, theta_0, radius):
                    """Project theta onto a ball of radius `radius` centered at `theta_0`."""
                    delta = theta - theta_0
                    norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True)
                    scaling_factor = torch.clamp(radius / (norm_delta + 1e-8), max=1.0)
                    if (scaling_factor < 1.0).any():
                        print("Projected!")
                    return theta_0 + scaling_factor * delta
            
                with torch.enable_grad():
                    # Compute the loss
                    alpha_t = pipe.scheduler.alphas_cumprod[pipe.scheduler.timesteps[i+1]]
                    loss = -0.5/(1 - alpha_t) * torch.norm(latents - torch.sqrt(alpha_t)*z0_pred_c) ** 2

                    gradients = torch.autograd.grad(loss, inputs=text_embeddings, retain_graph=False)[0]
            
                accumulated_grad += gradients/torch.norm(gradients, dim=-1, keepdim=True)
                accumulated_grad[0,:4] = 0

                # Free memory
                accumulated_grad = accumulated_grad.detach()  # Detach from graph
                
            else:
                text_embeddings = project_onto_ball(text_embeddings + 0.08*(max(0.9**(max(0,j - 10)), 0.001)) * accumulated_grad, text_embeddings_0, 15)

                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
                with torch.no_grad():
                    decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor ).sample.clip(-1, 1)
                    restored_x = (decoded_image / 2 + 0.5).clamp(0, 1)

                psnr = psnr_loss(restored_x, x_clean).item()
                ssim = ssim_loss(restored_x, x_clean).item()
                lpips = lpips_loss(restored_x * 2 - 1, x_clean * 2 -1).item()
                metrics = {
                    'PSNR' : psnr,
                    'SSIM' : ssim, 
                    'LPIPS': lpips
                }

                restored_x_lr = forward_model.A(restored_x.float())
                lr_psnr = psnr_loss(((y_norm+1)/2).clamp(0, 1), restored_x_lr).item()
                metrics['OBS-PSNR'] = lr_psnr

                metric_string = ""
                for m in metrics:
                    metric_string += f"{m}: {metrics[m]:.3f}, "
                print(metric_string)

                with open(csv_file, mode='a', newline='') as file:
                    # Write the metrics to the CSV file
                    writer = csv.writer(file)
                    writer.writerow([j + 1, psnr, ssim, lpips])

        if j<cfg.num_SAPG_steps-1:
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents2 = pipe.scheduler.add_noise(latents.detach(), noise=noise, timesteps=torch.tensor([999]))
            
                latents = latents2.clone()
                pipe.scheduler.set_timesteps(4, device=device)

                # Override the scheduler's timesteps with DMD2 values
                # custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
                custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
                pipe.scheduler.timesteps = custom_timesteps

                # Sample from the prior with the actual prompt c
                for _, timestep in enumerate(pipe.scheduler.timesteps):
                    noise_uncond = pipe.unet(
                        latents,
                        timestep,
                        encoder_hidden_states=text_embeddings,
                    ).sample
                    latents = pipe.scheduler.step(noise_uncond, timestep, latents).prev_sample

                # Decode latents to image
                decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

                save_image(torch.clamp(decoded_image * 0.5 + 0.5, 0, 1), os.path.join(xp_log_dir, f'prior_{j}.png'))
                # Prepare initial noise for next entry

                latents = latents2.clone()
                pipe.scheduler.set_timesteps(num_inference_steps, device=device)
                # Override the scheduler's timesteps with DMD2 values
                # custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
                custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
                pipe.scheduler.timesteps = custom_timesteps
        else:
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents2 = pipe.scheduler.add_noise(latents.detach(), noise=noise, timesteps=torch.tensor([999]))
            
                latents = latents2.clone()
                
                pipe.scheduler.set_timesteps(4, device=device)

                # Override the scheduler's timesteps with DMD2 values
                # custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
                custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
                pipe.scheduler.timesteps = custom_timesteps

                # Sample from the prior with the actual prompt c
                for k, timestep in enumerate(pipe.scheduler.timesteps):
                    noise_uncond = pipe.unet(
                        latents,
                        timestep,
                        encoder_hidden_states=text_embeddings,
                    ).sample
                    latents = pipe.scheduler.step(noise_uncond, timestep, latents).prev_sample

                with torch.no_grad():
                    # Decode latents to image
                    decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

                save_image(torch.clamp(decoded_image * 0.5 + 0.5, 0, 1), os.path.join(xp_log_dir, f'prior_{j}.png'))
                # Prepare initial noise for next entry

                latents = latents2.clone()
                del latents2
                
            pipe.scheduler.set_timesteps(8, device=device)
                
            # Override the scheduler's timesteps with DMD2 values
            custom_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long)
            #custom_timesteps = torch.tensor([999, 749, 499, 249], device=device, dtype=torch.long)
            pipe.scheduler.timesteps = custom_timesteps

            for i, timestep in enumerate(pipe.scheduler.timesteps):
                with torch.no_grad():
                    noise_uncond = pipe.unet(
                        latents,
                        timestep,
                        encoder_hidden_states=text_embeddings,
                    ).sample

                with torch.no_grad():
                    _, noise_pred = noise_pred_cond_y_PRO(
                        latents=latents,
                        t = timestep,
                        pipe=pipe,
                        cfg=cfg,
                        logdir=xp_log_dir,
                        y_guidance=y_norm,
                        forward_model=forward_model,
                        noise_pred=noise_uncond,
                        sigma_y = sigma_y_norm,
                        SAPG_j = j,
                        n_steps = 8
                    )
                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
                
    # End the timer
    end_time = time.time()

    # Print execution time
    print(f"Execution Time: {end_time - start_time:.6f} seconds")

    # Load the metrics from the CSV file
    csv_file = os.path.join(xp_log_dir, "metrics_log.csv")
    metrics_data = pd.read_csv(csv_file)

    # Define the figure size for each individual plot
    fig_size = (6, 5)

    # PSNR Plot
    plt.figure(figsize=fig_size)
    plt.plot(metrics_data['Iteration'], metrics_data['PSNR'], label='PSNR', color='b', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR')
    plt.title('PSNR Over Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(xp_log_dir, "PSNR_plot.png"), dpi=300)
    plt.close()

    # SSIM Plot
    plt.figure(figsize=fig_size)
    plt.plot(metrics_data['Iteration'], metrics_data['SSIM'], label='SSIM', color='g', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('SSIM')
    plt.title('SSIM Over Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(xp_log_dir, "SSIM_plot.png"), dpi=300)
    plt.close()

    # LPIPS Plot
    plt.figure(figsize=fig_size)
    plt.plot(metrics_data['Iteration'], metrics_data['LPIPS'], label='LPIPS', color='r', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('LPIPS')
    plt.title('LPIPS Over Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(xp_log_dir, "LPIPS_plot.png"), dpi=300)
    plt.close()

    
    with torch.no_grad():
        # Decode latents to image
        decoded_image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

    restored_x = (decoded_image / 2 + 0.5).clamp(0, 1)  # SAPG single-realization output

    save_image(restored_x, os.path.join(xp_log_dir, "restored.png"))
    save_image(((y_norm+1)/2).clamp(0, 1).detach().cpu(), os.path.join(xp_log_dir, "degraded.png"))
    save_image(x_clean.detach().cpu(), os.path.join(xp_log_dir, "clean.png"))

    # ------------------------------------------------------------------
    # Posterior sampling: run num_samples independent LCM passes using
    # the final optimised text_embeddings from SAPG.
    # ------------------------------------------------------------------
    num_samples = cfg.get('num_samples', 1)
    all_samples = []
    samples_dir = os.path.join(xp_log_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    sample_start = time.time()
    for sample_idx in range(num_samples):
        print(f"\n--- Posterior sample {sample_idx + 1}/{num_samples} ---")
        sample_seed = seed + sample_idx
        generator_s = torch.Generator(device=device).manual_seed(sample_seed)

        if cfg.init_strategy == 'y_noise' and mu_z is not None:
            noise_s = torch.randn(mu_z.shape, generator=generator_s, device=device, dtype=mu_z.dtype)
            latents_s = pipe.scheduler.add_noise(mu_z, noise=noise_s, timesteps=torch.tensor([999]))
        else:
            latents_s = pipe.prepare_latents(
                batch_size=1,
                num_channels_latents=pipe.unet.config.in_channels,
                height=image_height,
                width=image_width,
                dtype=torch.float16,
                device=device,
                generator=generator_s,
            )

        pipe.scheduler.set_timesteps(8, device=device)
        pipe.scheduler.timesteps = torch.tensor(
            [999, 874, 749, 624, 499, 374, 249, 124], device=device, dtype=torch.long
        )

        for i, timestep in enumerate(pipe.scheduler.timesteps):
            with torch.no_grad():
                noise_uncond_s = pipe.unet(
                    latents_s, timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample
                _, noise_pred_s = noise_pred_cond_y_PRO(
                    latents=latents_s, t=timestep, pipe=pipe, cfg=cfg,
                    logdir=xp_log_dir, y_guidance=y_norm, forward_model=forward_model,
                    noise_pred=noise_uncond_s, sigma_y=sigma_y_norm,
                    SAPG_j=cfg.num_SAPG_steps - 1, n_steps=8,
                )
                latents_s = pipe.scheduler.step(noise_pred_s, timestep, latents_s).prev_sample

        with torch.no_grad():
            decoded_s = pipe.vae.decode(latents_s / pipe.vae.config.scaling_factor).sample
        sample_x = (decoded_s / 2 + 0.5).clamp(0, 1)
        all_samples.append(sample_x.detach().cpu())
        save_image(sample_x, os.path.join(samples_dir, f'sample_{sample_idx:03d}.png'))
        print(f"Saved sample_{sample_idx:03d}.png")

    print(f"\nSampling time: {time.time() - sample_start:.2f}s")

    # Stack samples [N, C, H, W]
    all_samples_t = torch.cat(all_samples, dim=0)

    # Posterior mean and pixelwise std
    posterior_mean = all_samples_t.mean(0, keepdim=True)
    posterior_std  = all_samples_t.std(0, keepdim=True)
    save_image(posterior_mean, os.path.join(xp_log_dir, 'posterior_mean.png'))
    save_image(posterior_std.clamp(0, 1), os.path.join(xp_log_dir, 'posterior_std.png'))
    save_image((posterior_std * 5).clamp(0, 1), os.path.join(xp_log_dir, 'posterior_std_5x.png'))

    # Residuals
    x_clean_cpu = x_clean.detach().cpu()
    y_obs = y.clamp(0, 1).detach().cpu()
    with torch.no_grad():
        y_from_sample0 = forward_model.A(all_samples_t[0:1].to(device)).clamp(0, 1).detach().cpu()
        y_from_mean    = forward_model.A(posterior_mean.to(device)).clamp(0, 1).detach().cpu()

    save_image((x_clean_cpu - all_samples_t[0:1] + 1) / 2, os.path.join(xp_log_dir, 'residual_x_sample0.png'))
    save_image((y_obs - y_from_sample0 + 1) / 2,           os.path.join(xp_log_dir, 'residual_y_sample0.png'))
    save_image((x_clean_cpu - posterior_mean + 1) / 2,     os.path.join(xp_log_dir, 'residual_x_mean.png'))
    save_image((y_obs - y_from_mean + 1) / 2,              os.path.join(xp_log_dir, 'residual_y_mean.png'))

    # Final metrics computed on posterior mean
    posterior_mean_dev = posterior_mean.to(device)
    psnr  = psnr_loss(posterior_mean_dev, x_clean).item()
    ssim  = ssim_loss(posterior_mean_dev, x_clean).item()
    lpips = lpips_loss(posterior_mean_dev * 2 - 1, x_clean * 2 - 1).item()
    metrics = {
        'PSNR' : psnr,
        'SSIM' : ssim,
        'LPIPS': lpips,
    }

    if type(forward_model) == dinv.physics.Downsampling:
        restored_x_lr = forward_model.A(posterior_mean_dev.float())
        lr_psnr = psnr_loss(y_obs.to(device), restored_x_lr).item()
        metrics['OBS-PSNR'] = lr_psnr

    metric_string = ""
    for m in metrics:
        metric_string += f"{m}: {metrics[m]:.3f}, "
    print(metric_string)

    with open(os.path.join(xp_log_dir, 'metrics.csv'), 'w+') as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    main()

