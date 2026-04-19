# inverse_problems.py
import torch
from motionblur import Kernel
import deepinv as dinv

def get_forward_model(cfg, x_clean, device):
    noise_model = dinv.physics.GaussianNoise(sigma=cfg.problem.sigma_y)

    if cfg.problem.type == 'inpainting_squared_mask':
        B, C, H, W = x_clean.shape
        mask = torch.ones((1, H, W), device=x_clean.device)
        size = int(cfg.problem.mask_size)
        ref_res = int(cfg.problem.get("mask_reference_resolution", 1024))

        # The original mask formula was authored for 1024px images.
        # Scale it to the current image size so 512px runs use the same relative occlusion.
        scale_y = H / ref_res
        scale_x = W / ref_res
        size_y = max(1, round(size * scale_y))
        size_x = max(1, round(size * scale_x))
        offset_y = round(35 * scale_y)
        offset_x = round(2 * scale_x)

        top = max(0, min(H, H // 2 - size_y // 5 - offset_y))
        bottom = max(0, min(H, H // 2 + size_y // 5 - offset_y))
        left = max(0, min(W, W // 2 - 4 * size_x // 5 - offset_x))
        right = max(0, min(W, W // 2 + 4 * size_x // 5 + offset_x))

        if top >= bottom or left >= right:
            raise ValueError(
                f"Invalid inpainting mask bounds for image size {(H, W)} "
                f"with mask_size={size} and mask_reference_resolution={ref_res}."
            )

        mask[:, top:bottom, left:right] = 0
        # forward_model = dinv.physics.Inpainting(
        #     tensor_size=x_clean.shape,
        #     mask=mask,
        #     noise_model=noise_model,
        # ).to(device)
        
        # # 🔥 deepinv bug fix: mask is NOT moved by .to()
        # forward_model.mask = forward_model.mask.to(device)
        # transpose_operator = forward_model.A_adjoint
        # forward_model = forward_model.to(device)

        forward_model = dinv.physics.Inpainting(
            tensor_size=x_clean.shape,
            mask=mask,
            noise_model=noise_model,
        ).to(device)
        
        # 🔥 HARD FIX: ensure internal buffers are moved
        forward_model.to(device)
        
        # 🔥 CRITICAL: re-assign mask AFTER .to()
        forward_model.mask = forward_model.mask.to(device)

    elif cfg.problem.type == 'deblurring_gaussian':
        ksize = cfg.problem.sigma_kernel
        filter = dinv.physics.blur.gaussian_blur(sigma=(ksize, ksize))
        forward_model = dinv.physics.BlurFFT(
            img_size=x_clean.shape[1:],
            filter=filter,
            device=device,
            noise_model=noise_model,
        )
        transpose_operator = forward_model.A_adjoint

    elif cfg.problem.type == 'deblurring_motion':
        kernel = Kernel(size=(122, 122), intensity=0.5)
        kernel_torch = torch.tensor(kernel.kernelMatrix, dtype=torch.float32)
        kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(0).to(device)
        forward_model = dinv.physics.BlurFFT(
            img_size=x_clean.shape[1:],
            filter=kernel_torch,
            device=device,
            noise_model=noise_model,
        )
        transpose_operator = forward_model.A_adjoint

    elif cfg.problem.type == 'super_resolution_bicubic':
        forward_model = dinv.physics.Downsampling(
            img_size=x_clean.shape[1:],
            factor=cfg.problem.downscaling_factor,
            device=device,
            noise_model=noise_model,
            filter='bicubic',
            padding='reflect',
        )
        transpose_operator = forward_model.A_adjoint

    else:
        raise ValueError(f"Unexpected problem.type {cfg.problem.type}")

    return forward_model, transpose_operator
