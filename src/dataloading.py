import os
from glob import glob

import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import (
    Compose,
    RandomCrop,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToDtype,
    Normalize,
)
from torch.utils.data import Dataset


DEFAULT_TRANSFORM = Compose([
    RandomCrop(size=(256, 256)),
    RandomHorizontalFlip(p=0.5),
    # ToDtype(torch.float32, scale=True),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LOLImageDataset(Dataset):
    def __init__(self, root: str = 'data/LOL-v2/Real',
                 test: bool = True,
                 transform: Compose = DEFAULT_TRANSFORM) -> None:
        self.test: bool = test
        self.transform = transform

        self.images = self._find_images(root)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.images.iloc[idx]
        d = {k: read_image(v) for k, v in row.items()}
        # return d
        return self.transform(d)

    @property
    def par(self):
        return 'Test' if self.test else 'Train'

    def _find_images(self, root: str, mapping={'gt': 'Normal', 'lq': 'Low'}) -> pd.DataFrame:
        ret = pd.DataFrame()
        for alias, label in mapping.items():
            image_list = glob(f'{root}/{self.par}/{label}/*.png')
            updates = pd.Series(image_list,
                                index=[os.path.basename(x) for x in image_list])
            ret[alias] = updates
        return ret


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = LOLImageDataset(root='data/LOL-v2/Real', test=True)
    # (3, H, W)
    gt, lq = dataset[0].values()
    fig, axes = plt.subplots(1, 2)
    gt = gt.permute(1, 2, 0)
    print(gt.shape, lq.shape)
    axes[0].imshow(gt)
    axes[1].imshow(lq.permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
