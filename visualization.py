
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_visualization(lr_tensor, sr_tensor, hr_tensor, epoch, sample_index, output_dir, filename_prefix="vis"):
    os.makedirs(output_dir, exist_ok=True)

    lr_tensor = lr_tensor.cpu().detach()
    hr_tensor = hr_tensor.cpu().detach()
    sr_tensor = sr_tensor.cpu().detach()

    hr_img = hr_tensor.squeeze().numpy()
    sr_img = sr_tensor.squeeze().numpy()
    lr_img_original = lr_tensor.squeeze().numpy()

    hr_shape = (hr_img.shape[0], hr_img.shape[1])

    lr_img_upsampled_tensor = F.interpolate(
        lr_tensor.unsqueeze(0),
        size=hr_shape,
        mode='bicubic',
        align_corners=False
    )
    bicubic_img = lr_img_upsampled_tensor.squeeze().numpy()

    vmin, vmax = np.percentile(hr_img, [2, 98])

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))


    im1 = axes[0, 0].imshow(lr_img_original, cmap='terrain', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0, 0].set_title(f'LR\n{lr_img_original.shape[0]}x{lr_img_original.shape[1]}')
    axes[0, 0].axis('off')
    fig.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    im2 = axes[0, 1].imshow(bicubic_img, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Bicubic')
    axes[0, 1].axis('off')
    fig.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    im3 = axes[1, 0].imshow(sr_img, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('SR')
    axes[1, 0].axis('off')
    fig.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    im4 = axes[1, 1].imshow(hr_img, cmap='terrain', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Ground Truth')
    axes[1, 1].axis('off')
    fig.colorbar(im4, ax=axes[1, 1], shrink=0.8)


    plt.suptitle(f'Epoch {epoch} - Sample {sample_index}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_filename = f"{filename_prefix}_epoch_{epoch:03d}_sample_{sample_index:02d}.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)