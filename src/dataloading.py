import os
from glob import glob

import pandas as pd
import torch
from torch.nn import Parameter
from torchvision.io import read_image
from torchvision.tv_tensors import Image
from torchvision.transforms.v2 import (
    Transform,
    Compose,
    RandomCrop,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToDtype,
    Normalize,
)
from torch.utils.data import Dataset


class YUV2RGB(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.mat = Parameter(
            torch.tensor([[1.000, 0.000, 1.140],
                          [1.000, -0.395, -0.581],
                          [1.000, 2.032, 0.000]],
                         dtype=torch.float32),
            requires_grad=False
        )

    def __call__(self, x: Image) -> torch.Tensor:
        if type(x) == dict:
            return {k: self.forward(v) for k, v in x.items()}
        elif type(x) == Image or type(x) == torch.Tensor:
            return self.forward(x)
        elif type(x) == list or type(x) == tuple:
            return [self.forward(v) for v in x]

        assert False, f'no call script for type {type(x)}'

    def forward(self, x: Image) -> Image:
        v = torch.einsum('ij, ...jhw->...ihw', self.mat, x)
        return v.clamp(0, 1)


class RGB2YUV(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.mat = Parameter(
            torch.tensor([[0.299, 0.587, 0.114],
                          [-0.147, -0.289, 0.436],
                          [0.615, -0.515, -0.100]],
                         dtype=torch.float32),
            requires_grad=False
        )

    def __call__(self, x: Image | dict) -> Image | dict:
        if type(x) == dict:
            return {k: self.forward(v) for k, v in x.items()}
        elif type(x) == Image or type(x) == torch.Tensor:
            return self.forward(x)
        elif type(x) == list or type(x) == tuple:
            return [self.forward(v) for v in x]

        assert False, f'no call script for type {type(x)}'

    def forward(self, x: Image) -> Image:
        v = torch.einsum('ji, ...ihw->...jhw', self.mat, x)
        return v


RGB255_TO_YUV_TRANSFORM = Compose([
    ToDtype(torch.float32, scale=True),
    RGB2YUV(),
])


DEFAULT_TRANSFORM = Compose([
    RandomResizedCrop(size=(128, 128)),
    RandomHorizontalFlip(p=0.5),
    ToDtype(torch.float32, scale=True),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


CYANET_TRAIN_TF = Compose([
    RandomResizedCrop(size=(128, 128)),
    RandomHorizontalFlip(p=0.5),
    ToDtype(torch.float32, scale=True),
    RGB2YUV(),
])

CYANET_TEST_TF = Compose([
    ToDtype(torch.float32, scale=True),
    RGB2YUV(),
])


class LOLImageDataset(Dataset):
    def __init__(self, root: str,
                 partition: str = 'test',
                 transform: Compose = DEFAULT_TRANSFORM,
                 par_mapping: dict = {'test': 'test', 'train': 'train'},
                 gt_mapping: dict = {'gt': 'gt', 'lq': 'lq'}) -> None:
        self.partition = partition
        self.transform = transform

        self.images = self._find_images(
            f'{root}/{par_mapping[partition]}', gt_mapping)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Image]:
        row = self.images.iloc[idx]
        d = {k: Image(read_image(v)) for k, v in row.items()}

        if self.transform:
            d = self.transform(d)
        return d

    def _find_images(self, root: str, gt_mapping: dict) -> pd.DataFrame:
        ret = pd.DataFrame()
        for alias, label in gt_mapping.items():
            image_list = glob(f'{root}/{label}/*.png')
            updates = pd.Series(image_list,
                                index=[os.path.basename(x) for x in image_list])
            ret[alias] = updates
        assert ret.shape[0] > 0, f'No images found in {root}'
        return ret

    def peek(self, index: int = 0) -> Image:
        e = self[index]['gt']
        print(e.shape, e.min(), e.max())
        return e


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = LOLImageDataset('data/LOL', partition='test')
    fig, axes = plt.subplots(1, 2)

    # (3, H, W)
    sample = dataset[0]
    print(sample)
    gt = sample['gt'].permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    lq = sample['lq'].permute(1, 2, 0)
    print(gt.shape, lq.shape)
    axes[0].imshow(gt)
    axes[1].imshow(lq)
    plt.tight_layout()
    plt.show()
