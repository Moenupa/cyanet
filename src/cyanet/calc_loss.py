import torch
from torch.nn import Module, SmoothL1Loss, L1Loss

from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    MultiScaleStructuralSimilarityIndexMeasure as MSSSIM,
    StructuralSimilarityIndexMeasure as SSIM,
    LearnedPerceptualImagePatchSimilarity as LPIPS
)


class LossFn(Module):
    def __init__(self, alphas: list[int] = [1.00, 0.008, 0.05, 0.06, 0.25]) -> None:
        super().__init__()
        self.alphas = alphas

        self.pixel = SmoothL1Loss()
        self.psnr = PSNR(data_range=1.0)
        self.ssim = SSIM()
        self.lpips = LPIPS(net_type='vgg')
        self.color = L1Loss()

    def forward(self,
                rgb_gt: torch.Tensor, yuv_gt: torch.Tensor,
                rgb_pred: torch.Tensor, yuv_pred: torch.Tensor
                ) -> torch.Tensor:
        l_pixel = self.pixel(yuv_pred, yuv_gt)
        l_psnr = 50 - self.psnr(yuv_pred, yuv_gt)
        l_ssim = 1 - self.ssim(yuv_pred, yuv_gt)
        l_lpips = self.lpips(rgb_pred, rgb_gt)
        l_color = self.color(yuv_pred[:, 1:, :, :], yuv_gt[:, 1:, :, :])

        loss = self.alphas[0] * l_pixel + \
            self.alphas[1] * l_psnr + \
            self.alphas[2] * l_ssim + \
            self.alphas[3] * l_lpips + \
            self.alphas[4] * l_color

        return loss
