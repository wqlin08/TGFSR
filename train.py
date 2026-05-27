import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage import io
from tqdm import tqdm
import glob
import random
import math
import shutil



from TGFSR_AGEM_model import GeneratorSerialAGEM
from dem_features import Slope
from visualization import save_visualization
from loss  import  psd_loss

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class AverageMeter:
    def __init__(self): self.reset()

    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val;
        self.sum += val * n;
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class TXTLogger:

    def __init__(self, filepath, header, resume=False):
        self.filepath = filepath
        mode = 'a' if resume and os.path.exists(filepath) else 'w'
        self.file = open(filepath, mode)
        if mode == 'w':
            self.file.write(header + '\n')
            self.file.write('-' * (len(header) + 5) + '\n')
            self.file.flush()

    def log(self, log_str):
        self.file.write(log_str + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


class MultiLogger:


    def __init__(self, out_dir, resume=False):
        metrics_header = (
            f"{'Epoch':<7} | {'Train RMSE':<12} | {'Train MAE':<12} | {'Train PSNR':<12} | {'Train SSIM':<12} | "
            f"{'Val RMSE':<10} | {'Val MAE':<10} | {'Val PSNR':<10} | {'Val SSIM':<10}"
        )
        self.metrics_logger = TXTLogger(
            os.path.join(out_dir, 'log_metrics.txt'),
            metrics_header,
            resume
        )

        loss_header = (
            f"{'Epoch':<7} | {'Train Total':<12} | {'Train Content':<14} | {'Train PSD':<12} | {'Train Slope':<12} | "
            f"{'Val Total':<10} | {'Val Content':<12} | {'Val PSD':<10} | {'Val Slope':<10}"
        )
        self.loss_logger = TXTLogger(
            os.path.join(out_dir, 'log_losses.txt'),
            loss_header,
            resume
        )

    def log_metrics(self, data):

        log_str = (
            f"{data['epoch']:<7d} | "
            f"{data['train_rmse']:<12.4f} | {data['train_mae']:<12.4f} | {data['train_psnr']:<12.4f} | {data['train_ssim']:<12.4f} | "
            f"{data['val_rmse']:<10.4f} | {data['val_mae']:<10.4f} | {data['val_psnr']:<10.4f} | {data['val_ssim']:<10.4f}"
        )
        self.metrics_logger.log(log_str)

    def log_losses(self, data):
        log_str = (
            f"{data['epoch']:<7d} | "
            f"{data['train_total']:<12.4f} | {data['train_content']:<14.4f} | "
            f"{data['train_psd']:<12.4f} | {data['train_slope']:<12.4f} | "
            f"{data['val_total']:<10.4f} | {data['val_content']:<12.4f} | "
            f"{data['val_psd']:<10.4f} | {data['val_slope']:<10.4f}"
        )
        self.loss_logger.log(log_str)

    def close(self):
        self.metrics_logger.close()
        self.loss_logger.close()

class PairedDEMDataset(Dataset):
    def __init__(self, root_dir, use_augmentation=True):
        super(PairedDEMDataset, self).__init__()
        self.root_dir = root_dir
        self.use_augmentation = use_augmentation

        self.hr_dem_paths = sorted(glob.glob(os.path.join(root_dir, 'HR', '*.tif')))
        self.lr_dem_paths = sorted(glob.glob(os.path.join(root_dir, 'LR', '*.tif')))
        self.hr_sar_vv_paths = sorted(glob.glob(os.path.join(root_dir, 'SAR_VV', '*.tif')))
        self.hr_sar_vh_paths = sorted(glob.glob(os.path.join(root_dir, 'SAR_VH', '*.tif')))

        # Assertions to ensure data integrity
        assert len(self.hr_dem_paths) == len(self.lr_dem_paths), "HR and LR DEM counts do not match."
        assert len(self.hr_dem_paths) == len(self.hr_sar_vv_paths), "DEM and SAR VV counts do not match."
        assert len(self.hr_dem_paths) == len(self.hr_sar_vh_paths), "DEM and SAR VH counts do not match."
        assert len(self.hr_dem_paths) > 0, f"No data found in {root_dir}. Check paths."

    def __len__(self):
        return len(self.hr_dem_paths)

    def __getitem__(self, idx):
        hr_dem = io.imread(self.hr_dem_paths[idx]).astype(np.float32)
        lr_dem = io.imread(self.lr_dem_paths[idx]).astype(np.float32)
        hr_sar_vv = io.imread(self.hr_sar_vv_paths[idx]).astype(np.float32)
        hr_sar_vh = io.imread(self.hr_sar_vh_paths[idx]).astype(np.float32)

        if self.use_augmentation:
            if random.random() < 0.5:
                hr_dem, lr_dem = np.fliplr(hr_dem), np.fliplr(lr_dem)
                hr_sar_vv, hr_sar_vh = np.fliplr(hr_sar_vv), np.fliplr(hr_sar_vh)
            if random.random() < 0.5:
                hr_dem, lr_dem = np.flipud(hr_dem), np.flipud(lr_dem)
                hr_sar_vv, hr_sar_vh = np.flipud(hr_sar_vv), np.flipud(hr_sar_vh)
            k = random.randint(0, 3)
            if k > 0:
                hr_dem, lr_dem = np.rot90(hr_dem, k), np.rot90(lr_dem, k)
                hr_sar_vv, hr_sar_vh = np.rot90(hr_sar_vv, k), np.rot90(hr_sar_vh, k)

        hr_sar = np.stack([hr_sar_vv, hr_sar_vh], axis=0)

        return {
            'HR_DEM': torch.from_numpy(hr_dem.copy()).unsqueeze(0),
            'LR_DEM': torch.from_numpy(lr_dem.copy()).unsqueeze(0),
            'HR_SAR': torch.from_numpy(hr_sar.copy())
        }


def calculate_metrics_batch(fake_tensor, real_tensor):
    fake_img, real_img = fake_tensor.cpu().detach().numpy(), real_tensor.cpu().detach().numpy()
    mse, mae = np.mean((real_img - fake_img) ** 2), np.mean(np.abs(real_img - fake_img))
    psnr, ssim = 0, 0
    from skimage.metrics import structural_similarity
    for i in range(fake_img.shape[0]):
        real_sample, fake_sample = real_img[i, 0], fake_img[i, 0]
        data_range = real_sample.max() - real_sample.min()
        if data_range < 1e-6: data_range = 1.0
        mse_sample = np.mean((real_sample - fake_sample) ** 2)
        psnr_sample = 100.0 if mse_sample < 1e-10 else 20 * math.log10(data_range / math.sqrt(mse_sample))
        psnr += psnr_sample
        ssim += structural_similarity(real_sample, fake_sample, data_range=data_range)
    return {'mse': mse, 'mae': mae, 'psnr': psnr / fake_img.shape[0], 'ssim': ssim / fake_img.shape[0]}


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, opt):
    model.train()
    loss_meters = {'total': AverageMeter(), 'content': AverageMeter(), 'psd': AverageMeter(), 'slope': AverageMeter()}
    metric_meters = {'rmse': AverageMeter(), 'mae': AverageMeter(), 'psnr': AverageMeter(), 'ssim': AverageMeter()}
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{opt.nEpochs} [Training]")

    for batch in pbar:
        hr_dem_real = batch['HR_DEM'].to(device)
        lr_dem_input = batch['LR_DEM'].to(device)
        hr_sar_input = batch['HR_SAR'].to(device)

        batch_size = hr_dem_real.size(0)

        # --- Normalization ---
        b_min = lr_dem_input.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        b_max = lr_dem_input.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        range_val = b_max - b_min + 1e-6
        norm_hr_dem = 2 * (hr_dem_real - b_min) / range_val - 1
        norm_lr_dem = 2 * (lr_dem_input - b_min) / range_val - 1

        sar_min = hr_sar_input.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        sar_max = hr_sar_input.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        sar_range = sar_max - sar_min + 1e-6
        norm_hr_sar = 2 * (hr_sar_input - sar_min) / sar_range - 1

        norm_hr_fake = model(norm_lr_dem, norm_hr_sar)

        # --- Loss Calculation ---
        content_loss = criterion['content'](norm_hr_fake, norm_hr_dem)

        freq_loss = opt.psdWeight * criterion['psd'](norm_hr_fake, norm_hr_dem)

        target_slope, output_slope = criterion['slope_net'](norm_hr_dem), criterion['slope_net'](norm_hr_fake)
        slope_loss = opt.slopeWeight * criterion['content'](output_slope, target_slope)

        total_loss = content_loss + freq_loss + slope_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            hr_fake = 0.5 * (norm_hr_fake.detach() + 1) * range_val + b_min
            loss_meters['total'].update(total_loss.item(), batch_size)
            loss_meters['content'].update(content_loss.item(), batch_size)
            loss_meters['psd'].update(freq_loss.item(), batch_size)
            loss_meters['slope'].update(slope_loss.item(), batch_size)

            metrics = calculate_metrics_batch(hr_fake, hr_dem_real)
            metric_meters['rmse'].update(math.sqrt(metrics['mse']), batch_size)
            metric_meters['mae'].update(metrics['mae'], batch_size)
            metric_meters['psnr'].update(metrics['psnr'], batch_size)
            metric_meters['ssim'].update(metrics['ssim'], batch_size)

        pbar.set_postfix(Loss=f"{loss_meters['total'].avg:.4f}", RMSE=f"{metric_meters['rmse'].avg:.4f}")

    return {k: v.avg for k, v in loss_meters.items()}, {k: v.avg for k, v in metric_meters.items()}


def validate_one_epoch(model, loader, criterion, device, epoch, opt):
    """
    Validate the model over one cycle and calculate all performance metrics and loss values.

    Args:
        model (nn.Module): The model to be validated.
        loader (DataLoader): The data loader for the validation set.
        criterion (dict): A dictionary containing all loss functions.
        device (torch.device): The computing device (CPU or CUDA).
        epoch (int): The current epoch number, used for logging and visualization.
        opt (argparse.Namespace)`: An object containing all command-line arguments.

    Returns:
        tuple[dict, dict]: A dictionary containing average loss values ​​and a dictionary containing average performance metrics.
    """
    model.eval()

    loss_meters = {
        'total': AverageMeter(),
        'content': AverageMeter(),
        'psd': AverageMeter(),
        'slope': AverageMeter()
    }
    metric_meters = {
        'rmse': AverageMeter(),
        'mae': AverageMeter(),
        'psnr': AverageMeter(),
        'ssim': AverageMeter()
    }

    vis_batch_idx = random.randint(0, len(loader) - 1) if len(loader) > 0 else -1

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{opt.nEpochs} [Validation]")
        for i, batch in enumerate(pbar):

            hr_dem_real = batch['HR_DEM'].to(device)
            lr_dem_input = batch['LR_DEM'].to(device)
            hr_sar_input = batch['HR_SAR'].to(device)
            batch_size = hr_dem_real.size(0)

            b_min = lr_dem_input.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
            b_max = lr_dem_input.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
            range_val = b_max - b_min + 1e-6
            norm_hr_dem = 2 * (hr_dem_real - b_min) / range_val - 1
            norm_lr_dem = 2 * (lr_dem_input - b_min) / range_val - 1


            sar_min = hr_sar_input.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
            sar_max = hr_sar_input.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
            sar_range = sar_max - sar_min + 1e-6
            norm_hr_sar = 2 * (hr_sar_input - sar_min) / sar_range - 1

            norm_hr_fake = model(norm_lr_dem, norm_hr_sar)

            content_loss = criterion['content'](norm_hr_fake, norm_hr_dem)
            freq_loss = opt.psdWeight * criterion['psd'](norm_hr_fake, norm_hr_dem)
            target_slope = criterion['slope_net'](norm_hr_dem)
            output_slope = criterion['slope_net'](norm_hr_fake)
            slope_loss = opt.slopeWeight * criterion['content'](output_slope, target_slope)
            total_loss = content_loss + freq_loss + slope_loss

            loss_meters['total'].update(total_loss.item(), batch_size)
            loss_meters['content'].update(content_loss.item(), batch_size)
            loss_meters['psd'].update(freq_loss.item(), batch_size)
            loss_meters['slope'].update(slope_loss.item(), batch_size)

            hr_fake = 0.5 * (norm_hr_fake + 1) * range_val + b_min
            metrics = calculate_metrics_batch(hr_fake, hr_dem_real)

            metric_meters['rmse'].update(math.sqrt(metrics['mse']), batch_size)
            metric_meters['mae'].update(metrics['mae'], batch_size)
            metric_meters['psnr'].update(metrics['psnr'], batch_size)
            metric_meters['ssim'].update(metrics['ssim'], batch_size)

            pbar.set_postfix(RMSE=f"{metric_meters['rmse'].avg:.4f}", Val_Loss=f"{loss_meters['total'].avg:.4f}")

            if i == vis_batch_idx:
                vis_dir = os.path.join(opt.out, 'visualizations')
                for j in range(min(opt.vis_samples, batch_size)):
                    save_visualization(
                        lr_tensor=lr_dem_input[j],
                        sr_tensor=hr_fake[j],
                        hr_tensor=hr_dem_real[j],
                        epoch=epoch,
                        sample_index=j + 1,
                        output_dir=vis_dir
                    )

    avg_losses = {key: meter.avg for key, meter in loss_meters.items()}
    avg_metrics = {key: meter.avg for key, meter in metric_meters.items()}

    return avg_losses, avg_metrics


# ------------------- Main Execution Block -------------------
def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(opt.out, exist_ok=True)
    os.makedirs(os.path.join(opt.out, 'visualizations'), exist_ok=True)

    log_filepath = os.path.join(opt.out, 'training_log.txt')
    logger = MultiLogger(opt.out, resume=(opt.resume_weight or opt.auto_resume))

    use_augmentation = (opt.augment.lower() == 'true')
    print(f"Data augmentation is {'ENABLED' if use_augmentation else 'DISABLED'} for the training set.")

    train_dataset = PairedDEMDataset(opt.dataroot, use_augmentation=use_augmentation)
    val_dataset = PairedDEMDataset(opt.val_dataroot, use_augmentation=False)  # Validation set never uses augmentation

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False,
                            num_workers=opt.workers, pin_memory=True)

    print(f"Training data: {len(train_loader.dataset)} samples | Validation data: {len(val_loader.dataset)} samples")

    model = GeneratorSerialAGEM(
        n_res_blocks=opt.n_res_blocks,
        n_agem_blocks=opt.n_agem_blocks,
        upsample_factor=opt.upSampling,
        agem_iterations=opt.agem_iterations
    ).to(device)
    print(f"Model created: {opt.n_res_blocks} ResBlocks, {opt.n_agem_blocks} AGEM blocks with {opt.agem_iterations} iterations each.")

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nEpochs, eta_min=opt.eta_min)

    criterion = {
        'content': nn.L1Loss().to(device),
        'slope_net': Slope().to(device),
        'psd': psd_loss
    }
    for param in criterion['slope_net'].parameters(): param.requires_grad = False

    start_epoch = 0
    best_val_rmse = float('inf')
    resume_path = opt.resume_weight
    if not resume_path and opt.auto_resume:
        latest_path = os.path.join(opt.out, 'latest_model.pth')
        if os.path.exists(latest_path):
            resume_path = latest_path
            print(f"Auto-resuming from '{latest_path}'")

    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        print(f"Resumed successfully. Starting from epoch {start_epoch + 1}. Best historical RMSE: {best_val_rmse:.4f}")

    for epoch in range(start_epoch, opt.nEpochs):
        current_epoch = epoch + 1

        train_losses, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, current_epoch,
                                                      opt)
        scheduler.step()

        val_losses, val_metrics = validate_one_epoch(model, val_loader, criterion, device, current_epoch, opt)

        metrics_data = {
            'epoch': current_epoch,
            'train_rmse': train_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'train_psnr': train_metrics['psnr'],
            'train_ssim': train_metrics['ssim'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_psnr': val_metrics['psnr'],
            'val_ssim': val_metrics['ssim'],
        }
        logger.log_metrics(metrics_data)

        losses_data = {
            'epoch': current_epoch,
            # Training losses
            'train_total': train_losses['total'],
            'train_content': train_losses['content'],
            'train_psd': train_losses['psd'],
            'train_slope': train_losses['slope'],
            # Validation losses
            'val_total': val_losses['total'],
            'val_content': val_losses['content'],
            'val_psd': val_losses['psd'],
            'val_slope': val_losses['slope'],
        }
        logger.log_losses(losses_data)

        console_log = (
            f"\n--- Epoch {current_epoch}/{opt.nEpochs} Summary ---\n"
            f"  Train | Loss: {train_losses['total']:.4f}, RMSE: {train_metrics['rmse']:.4f}, PSNR: {train_metrics['psnr']:.2f}\n"
            f"  Val   | Loss: {val_losses['total']:.4f}, RMSE: {val_metrics['rmse']:.4f}, PSNR: {val_metrics['psnr']:.2f}\n"
            f"  LR: {scheduler.get_last_lr()[0]:.6f}\n"
            f"---------------------------------------\n"
        )
        print(console_log)

        is_best = val_metrics['rmse'] < best_val_rmse
        if is_best:
            best_val_rmse = val_metrics['rmse']
            print(f"*** New best model found! Val RMSE: {best_val_rmse:.4f}. Saved to best_rmse_model.pth ***")
            torch.save(model.state_dict(), os.path.join(opt.out, 'best_rmse_model.pth'))

        checkpoint_state = {
            'epoch': current_epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_val_rmse': best_val_rmse
        }
        torch.save(checkpoint_state, os.path.join(opt.out, 'latest_model.pth'))

        if (current_epoch) % opt.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(opt.out, f'generator_{current_epoch:03d}.pth'))

    logger.close()
    print(f"Training complete! Log saved to {log_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- Data and training arguments ---
    parser.add_argument('--dataroot', type=str, default=r'path/to/your/dataset/train',
                        help='path to training dataset')
    parser.add_argument('--val_dataroot', type=str, default=r'path/to/your/dataset/val',
                        help='path to validation dataset')
    parser.add_argument('--imageSize', type=int, default=72, help='size of the high-resolution images')
    parser.add_argument('--upSampling', type=int, default=4, help='super resolution upsampling factor')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--eta_min', type=float, default=5e-7, help='Minimum learning rate of cosine annealing scheduler')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for optimizer')
    parser.add_argument('--augment', type=str, default='false', choices=['true', 'false'],
                        help='enable or disable data augmentation for training set')

    parser.add_argument('--n_res_blocks', type=int, default=1, help='number of residual blocks in LR path')
    parser.add_argument('--n_agem_blocks', type=int, default=1, help='number of AGEM blocks in MR path')

    parser.add_argument('--agem_iterations', type=int, default=3, help='number of iterations within each AGEM block')

    parser.add_argument('--psdWeight', type=float, default=0, help='weight for the PSD loss component, the default setting is 0, which yields better results.')
    parser.add_argument('--slopeWeight', type=float, default=0.2, help='weight for the slope loss component')

    parser.add_argument('--out', default=r'path/to/your/results/train',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--resume_weight', type=str, default="", help='path to checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true',
                        help='automatically resume from latest_model.pth in the output directory')
    parser.add_argument('--save_interval', type=int, default=5, help='save a numbered checkpoint every N epochs')
    parser.add_argument('--vis_samples', type=int, default=4, help='number of samples to save for visualization')

    opt = parser.parse_args()

    main(opt)
