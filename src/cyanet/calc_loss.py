import torch
from torch.nn import Module, SmoothL1Loss
from src.dataloading import YUV2RGB

from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    MultiScaleStructuralSimilarityIndexMeasure as MSSSIM,
    StructuralSimilarityIndexMeasure as SSIM,
    LearnedPerceptualImagePatchSimilarity as LPIPS
)


class LossFn(Module):
    def __init__(self, alphas: list[int] = [1.00, 0.008, 0.5, 0.06]) -> None:
        super().__init__()
        self.alphas = alphas

        self.pixel = SmoothL1Loss()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS(net_type='vgg')
        self.rgb = YUV2RGB()

    def forward(self,
                gt: torch.Tensor,
                pred: torch.Tensor
                ) -> torch.Tensor:
        rgb_gt = self.rgb(gt)
        rgb_pred = self.rgb(pred)
        l_pixel = self.pixel(pred, gt)
        l_psnr = 50.0 - self.psnr(pred, gt)
        l_ssim = 1.0 - self.ssim(pred, gt)
        l_lpips = self.lpips(rgb_pred, rgb_gt)

        loss = self.alphas[0] * l_pixel + \
            self.alphas[1] * l_psnr + \
            self.alphas[2] * l_ssim + \
            self.alphas[3] * l_lpips

        return loss, l_pixel, l_psnr, l_ssim, l_lpips
