import torch
from torch.nn import Module, SmoothL1Loss

from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    MultiScaleStructuralSimilarityIndexMeasure as MSSSIM,
    StructuralSimilarityIndexMeasure as SSIM,
    LearnedPerceptualImagePatchSimilarity as LPIPS
)


class LossFn(Module):
    def __init__(self, alphas: list[int] = [1.00, 0.008, 0.05, 0.06]) -> None:
        super().__init__()
        self.alphas = alphas

        self.l1 = SmoothL1Loss(reduction='mean')
        self.psnr = PSNR(data_range=1.0)
        self.ssim = SSIM()
        self.lpips = LPIPS(net_type='vgg')

    def forward(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        loss = self.alphas[0] * self.l1(pred, gt) + \
            self.alphas[1] * (40 - self.psnr(pred, gt)) + \
            self.alphas[2] * self.ssim(pred, gt) + \
            self.alphas[3] * self.lpips(pred, gt)

        return loss
