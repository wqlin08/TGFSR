import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def swish(x): return x * torch.sigmoid(x)


class AGEM(nn.Module):
    """
    Attribute-Guided Iterative Evolution Module
    """
    def __init__(self, in_channels, num_iterations=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_iterations = num_iterations
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.terrain_to_flow_preference = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, 8, 1)
        )
        self.flow_update_cell = nn.Sequential(
            nn.Conv2d(8 + 8, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 8, 1)
        )
        self.carver_builder = nn.Conv2d(in_channels, in_channels * 2, 3, padding=1)
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, 1)

    def calculate_inflow(self, flow_weights):
        accumulated = torch.zeros_like(flow_weights[:, 0:1, :, :])
        accumulated[:, :, 1:, 1:] += flow_weights[:, 7, :-1, :-1].unsqueeze(1)
        accumulated[:, :, 1:, :] += flow_weights[:, 6, :-1, :].unsqueeze(1)
        accumulated[:, :, 1:, :-1] += flow_weights[:, 5, :-1, 1:].unsqueeze(1)
        accumulated[:, :, :, 1:] += flow_weights[:, 4, :, :-1].unsqueeze(1)
        accumulated[:, :, :, :-1] += flow_weights[:, 3, :, 1:].unsqueeze(1)
        accumulated[:, :, :-1, 1:] += flow_weights[:, 2, 1:, :-1].unsqueeze(1)
        accumulated[:, :, :-1, :] += flow_weights[:, 1, 1:, :].unsqueeze(1)
        accumulated[:, :, :-1, :-1] += flow_weights[:, 0, 1:, 1:].unsqueeze(1)
        return accumulated

    def forward(self, x):
        x_for_grad = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(x_for_grad, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_for_grad, self.sobel_y, padding=1)
        slope = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        aspect_cos = grad_x / (slope + 1e-6)
        aspect_sin = grad_y / (slope + 1e-6)
        terrain_attributes_single = torch.cat([x_for_grad, slope, aspect_cos, aspect_sin], dim=1)
        terrain_attributes = terrain_attributes_single.repeat(1, self.in_channels, 1, 1)
        flow_preference_logits = self.terrain_to_flow_preference(terrain_attributes)
        current_flow_logits = flow_preference_logits
        for _ in range(self.num_iterations):
            current_flow_weights = torch.softmax(current_flow_logits, dim=1)
            update_input = torch.cat([current_flow_weights, flow_preference_logits], dim=1)
            update_delta = self.flow_update_cell(update_input)
            current_flow_logits = current_flow_logits + update_delta
        final_flow_weights = torch.softmax(current_flow_logits, dim=1)
        convergence_map = torch.sigmoid(self.calculate_inflow(final_flow_weights))
        entropy_map = -torch.sum(final_flow_weights * torch.log(final_flow_weights + 1e-8), dim=1,
                                 keepdim=True) / math.log(8.0)
        potential_features = self.carver_builder(x)
        potential_gully, potential_ridge = torch.chunk(potential_features, 2, dim=1)
        final_gully = convergence_map * potential_gully
        final_ridge = entropy_map * potential_ridge
        combined = torch.cat([x, final_gully, final_ridge], dim=1)
        fused_output = self.fusion_conv(combined)
        return x + fused_output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResidualBlockCBAM(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(ResidualBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)
        self.cbam = CBAM(n)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))  # Using standard ReLU
        y = self.bn2(self.conv2(y))
        y = self.cbam(y)
        return y + x


class FPNFusion(nn.Module):
    def __init__(self, in_channels):
        super(FPNFusion, self).__init__()
        self.lateral_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.smooth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(in_channels)

    def forward(self, x_lateral, x_top):
        # x_top is higher-res, so downsample it
        x_top = F.interpolate(x_top, size=x_lateral.shape[2:], mode='bilinear', align_corners=False)
        x_lateral = self.lateral_conv(x_lateral)
        out = x_lateral + x_top
        out = self.smooth_conv(out)
        out = self.cbam(out)
        return out


class ProgressiveUpsample(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(ProgressiveUpsample, self).__init__()
        self.steps = int(math.log2(scale_factor))
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        ) for _ in range(self.steps)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SAREncoder(nn.Module):
    def __init__(self, in_channels=2, num_features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features // 2, 3, padding=1),
            nn.BatchNorm2d(num_features // 2), nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, num_features, 3, padding=1)
        )

    def forward(self, hr_sar):
        return self.encoder(hr_sar)


class SarDemFusionModule(nn.Module):
    """
    This module is responsible for processing SAR features and fusing them with DEM features.

    This module includes KPN.
    """

    def __init__(self, num_features=64):
        super().__init__()
        self.sar_encoder = SAREncoder(in_channels=2, num_features=num_features)

        self.kpn = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=2, dilation=2),
            nn.BatchNorm2d(num_features), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.Tanh()
        )

        self.residual_conv = nn.Conv2d(num_features, 1, 3, padding=1)

    def forward(self, hr_dem_feat, hr_sar_input):

        hr_sar_feat = self.sar_encoder(hr_sar_input)

        modulation_map = self.kpn(hr_sar_feat)

        modulated_feat = hr_dem_feat * (1 + modulation_map)

        residual = self.residual_conv(modulated_feat)
        return residual


class GeneratorSerialAGEM(nn.Module):
    def __init__(self, n_res_blocks, n_agem_blocks, upsample_factor, num_features=64, agem_iterations=3):

        super(GeneratorSerialAGEM, self).__init__()
        if upsample_factor != 4:
            raise ValueError("This architecture is designed for an upsample_factor of 4.")
        self.upsample_factor = upsample_factor

        self.conv_first = nn.Conv2d(1, num_features, 3, padding=1)
        self.res_blocks = nn.ModuleList(
            [ResidualBlockCBAM(in_channels=num_features) for _ in range(n_res_blocks)]
        )
        self.conv_after_res = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.upsample_to_mr = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.agem_blocks = nn.ModuleList(
            [AGEM(in_channels=num_features, num_iterations=agem_iterations) for _ in range(n_agem_blocks)]
        )
        self.conv_after_agem = nn.Conv2d(num_features, num_features, 3, padding=1)

        self.upsample_to_hr = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.fusion_module = SarDemFusionModule(num_features=num_features)

    def forward(self, x_lr_dem, x_hr_sar):

        feat_shallow = self.conv_first(x_lr_dem)
        feat_body = feat_shallow
        for block in self.res_blocks:
            feat_body = block(feat_body)
        lr_feat = feat_shallow + self.conv_after_res(feat_body)

        mr_feat_base = self.upsample_to_mr(lr_feat)
        feat_body = mr_feat_base
        for block in self.agem_blocks:
            feat_body = block(feat_body)
        mr_feat_refined = mr_feat_base + self.conv_after_agem(feat_body)

        hr_dem_feat = self.upsample_to_hr(mr_feat_refined)

        dynamic_residual = self.fusion_module(hr_dem_feat, x_hr_sar)

        bicubic_baseline = F.interpolate(x_lr_dem, scale_factor=self.upsample_factor, mode='bicubic',
                                         align_corners=False)
        output = bicubic_baseline + dynamic_residual

        return output