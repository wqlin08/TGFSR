import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity
from tqdm import tqdm
import glob
import math
from osgeo import gdal


from TGFSR_AGEM_model import GeneratorSerialAGEM


class TestDEMDataset(Dataset):


    def __init__(self, root_dir):
        super(TestDEMDataset, self).__init__()
        self.root_dir = root_dir

        self.hr_dem_paths = sorted(glob.glob(os.path.join(root_dir, 'HR', '*.tif')))
        self.lr_dem_paths = sorted(glob.glob(os.path.join(root_dir, 'LR', '*.tif')))
        self.hr_sar_vv_paths = sorted(glob.glob(os.path.join(root_dir, 'SAR_VV', '*.tif')))
        self.hr_sar_vh_paths = sorted(glob.glob(os.path.join(root_dir, 'SAR_VH', '*.tif')))

        assert len(self.hr_dem_paths) > 0, f"No data found in {root_dir}. Please check the path."
        assert len(self.hr_dem_paths) == len(self.lr_dem_paths)
        assert len(self.hr_dem_paths) == len(self.hr_sar_vv_paths)
        assert len(self.hr_dem_paths) == len(self.hr_sar_vh_paths)

    def __len__(self):
        return len(self.hr_dem_paths)

    def __getitem__(self, idx):
        hr_dem_path = self.hr_dem_paths[idx]
        hr_dem = gdal.Open(hr_dem_path).ReadAsArray().astype(np.float32)
        lr_dem = gdal.Open(self.lr_dem_paths[idx]).ReadAsArray().astype(np.float32)
        hr_sar_vv = gdal.Open(self.hr_sar_vv_paths[idx]).ReadAsArray().astype(np.float32)
        hr_sar_vh = gdal.Open(self.hr_sar_vh_paths[idx]).ReadAsArray().astype(np.float32)

        # Stack SAR channels into a (2, H, W) tensor.
        hr_sar = np.stack([hr_sar_vv, hr_sar_vh], axis=0)

        filename = os.path.basename(hr_dem_path)

        return {
            'HR_DEM': torch.from_numpy(hr_dem.copy()).unsqueeze(0),
            'LR_DEM': torch.from_numpy(lr_dem.copy()).unsqueeze(0),
            'HR_SAR': torch.from_numpy(hr_sar.copy()),
            'filename': filename,
            'hr_dem_path': hr_dem_path  # Keep the source path as the GDAL reference.
        }


def save_geotiff(array, output_path, reference_tif_path):

    ref_ds = gdal.Open(reference_tif_path)
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()

    height, width = array.shape

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)

    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)

    out_band.FlushCache()
    out_ds = None
    ref_ds = None


def calculate_metrics_single(fake_img, real_img):

    mse = np.mean((real_img - fake_img) ** 2)
    mae = np.mean(np.abs(real_img - fake_img))
    rmse = math.sqrt(mse)

    data_range = real_img.max() - real_img.min()
    if data_range < 1e-6:
        data_range = 1.0  # Avoid division by zero for flat images.

    # Calculate PSNR.
    psnr = 100.0 if mse < 1e-10 else 20 * math.log10(data_range / rmse)

    # Calculate SSIM.
    ssim = structural_similarity(real_img, fake_img, data_range=data_range)

    return {'rmse': rmse, 'mae': mae, 'psnr': psnr, 'ssim': ssim}


def test(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(opt.output_dir, exist_ok=True)
    images_save_dir = os.path.join(opt.output_dir, 'SR_images_tif')
    os.makedirs(images_save_dir, exist_ok=True)

    print(f"Load test data from the directory:{opt.test_dir}")
    test_dataset = TestDEMDataset(opt.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Load the model from the path: {opt.model_path}")
    model = GeneratorSerialAGEM(
        n_res_blocks=opt.n_res_blocks,
        n_agem_blocks=opt.n_agem_blocks,
        upsample_factor=opt.upsample_factor
    ).to(device)

    try:
        model.load_state_dict(torch.load(opt.model_path, map_location=device))
    except (RuntimeError, KeyError):
        checkpoint = torch.load(opt.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("The model state was successfully loaded from the training checkpoint.")

    model.eval()

    all_metrics = {'rmse': [], 'mae': [], 'psnr': [], 'ssim': []}
    results_filepath = os.path.join(opt.output_dir, 'test_results.txt')

    with open(results_filepath, 'w', encoding='utf-8') as f_results:
        f_results.write(f"{'Filename':<30} | {'RMSE':<10} | {'MAE':<10} | {'PSNR':<10} | {'SSIM':<10}\n")
        f_results.write('-' * 75 + '\n')

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="test...")
            for batch in pbar:
                # --- Data Preparation ---
                hr_dem_real = batch['HR_DEM'].to(device)
                lr_dem_input = batch['LR_DEM'].to(device)
                hr_sar_input = batch['HR_SAR'].to(device)
                filename = batch['filename'][0]
                hr_dem_path = batch['hr_dem_path'][0]

                b_min = lr_dem_input.min()
                b_max = lr_dem_input.max()
                range_val = b_max - b_min + 1e-6
                norm_lr_dem = 2 * (lr_dem_input - b_min) / range_val - 1

                sar_min = hr_sar_input.min()
                sar_max = hr_sar_input.max()
                sar_range = sar_max - sar_min + 1e-6
                norm_hr_sar = 2 * (hr_sar_input - sar_min) / sar_range - 1

                # --- Model Inference ---
                norm_hr_fake = model(norm_lr_dem, norm_hr_sar)

                # --- Denormalization ---
                hr_fake = 0.5 * (norm_hr_fake + 1) * range_val + b_min

                # --- Metric Calculation ---
                sr_dem_np = hr_fake.squeeze().cpu().numpy()
                gt_dem_np = hr_dem_real.squeeze().cpu().numpy()

                metrics = calculate_metrics_single(sr_dem_np, gt_dem_np)
                for key in all_metrics.keys():
                    all_metrics[key].append(metrics[key])

                # --- Save Results ---
                log_str = (
                    f"{filename:<30} | "
                    f"{metrics['rmse']:<10.4f} | "
                    f"{metrics['mae']:<10.4f} | "
                    f"{metrics['psnr']:<10.4f} | "
                    f"{metrics['ssim']:<10.4f}\n"
                )
                f_results.write(log_str)
                f_results.flush()

                save_path = os.path.join(images_save_dir, f'{filename}')
                save_geotiff(sr_dem_np.astype(np.float32), save_path, hr_dem_path)

    avg_rmse = np.mean(all_metrics['rmse'])
    avg_mae = np.mean(all_metrics['mae'])
    avg_psnr = np.mean(all_metrics['psnr'])
    avg_ssim = np.mean(all_metrics['ssim'])

    summary = (
        f"\n--- Test Summary ---\n"
        f"  Total sample size: {len(test_dataset)}\n"
        f"  Average RMSE: {avg_rmse:.4f}\n"
        f"  Average MAE:  {avg_mae:.4f}\n"
        f"  Average PSNR: {avg_psnr:.4f} dB\n"
        f"  Average SSIM: {avg_ssim:.4f}\n"
        f"----------------------\n"
        f"Detailed indicators have been saved to:{results_filepath}\n"
        f"The generated GeoTIFF image has been saved to: {images_save_dir}\n"
    )
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TGFSR TEST Script")


    parser.add_argument('--model_path', type=str, default=r"path/to/your/results/train/best_rmse_model.pth",
                        help='The path to the trained generator model .pth file (required).')
    parser.add_argument('--test_dir', type=str, default=r'path/to/your/dataset/test',
                        help='The root directory path of the test dataset.')
    parser.add_argument('--output_dir', type=str, default=r'path/to/your/results/test',
                        help='A directory used to save the results (indicator files and generated images).')

    parser.add_argument('--n_res_blocks', type=int, default=1,
                        help='The number of residual blocks in the LR path.')
    parser.add_argument('--n_agem_blocks', type=int, default=1,
                        help='The number of AGEM blocks in the MR path.')
    parser.add_argument('--upsample_factor', type=int, default=4,
                        help='Super-resolution upsampling factor.')

    opt = parser.parse_args()

    test(opt)

