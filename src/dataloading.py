import os
from glob import glob

import pandas as pd
import torch
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


DEBUG = False


MAT_YUV2RGB = torch.tensor([[1.000, 0.000, 1.140],
                            [1.000, -0.395, -0.581],
                            [1.000, 2.032, 0.000]],
                           dtype=torch.float32)

MAT_RGB2YUV = torch.tensor([[0.299, 0.587, 0.114],
                            [-0.147, -0.289, 0.436],
                            [0.615, -0.515, -0.100]],
                           dtype=torch.float32)


class YUV2RGB(Transform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Image) -> torch.Tensor:
        if type(x) == dict:
            return {k: self.yuv2rgb(v) for k, v in x.items()}
        elif type(x) == Image or type(x) == torch.Tensor:
            return self.yuv2rgb(x)
        elif type(x) == list or type(x) == tuple:
            return [self.yuv2rgb(v) for v in x]

        assert False, f'no call script for type {type(x)}'

    @staticmethod
    def yuv2rgb(x: Image) -> Image:
        v = torch.einsum('ij, jhw->ihw', MAT_YUV2RGB, x)
        if DEBUG:
            print('r:', v[0, :, :].min(), v[0, :, :].max(),
                  'g:', v[1, :, :].min(), v[1, :, :].max(),
                  'b:', v[2, :, :].min(), v[2, :, :].max())
        return v.clamp(0, 1)


class RGB2YUV(Transform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Image | dict) -> Image | dict:
        if type(x) == dict:
            return {k: self.rgb2yuv(v) for k, v in x.items()}
        elif type(x) == Image or type(x) == torch.Tensor:
            return self.rgb2yuv(x)
        elif type(x) == list or type(x) == tuple:
            return [self.rgb2yuv(v) for v in x]

        assert False, f'no call script for type {type(x)}'

    @staticmethod
    def rgb2yuv(x: Image) -> Image:
        v = torch.einsum('ji, ihw->jhw', MAT_RGB2YUV, x)
        if DEBUG:
            print('y:', v[0, :, :].min(), v[0, :, :].max(),
                  'u:', v[1, :, :].min(), v[1, :, :].max(),
                  'v:', v[2, :, :].min(), v[2, :, :].max())
        return v


RGB255_TO_YUV_TRANSFORM = Compose([
    ToDtype(torch.float32, scale=True),
    RGB2YUV(),
])


DEFAULT_TRANSFORM = Compose([
    RandomCrop(size=(256, 256)),
    RandomHorizontalFlip(p=0.5),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    dataset = LOLImageDataset(root='data/LOL/',
                              partition='test')
    # (3, H, W)
    gt, lq = dataset[0].values()
    fig, axes = plt.subplots(1, 2)
    gt = gt.permute(1, 2, 0)
    print(gt.shape, lq.shape)
    axes[0].imshow(gt)
    axes[1].imshow(lq.permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
