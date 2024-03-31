import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from dataloading import LOLImageDataset
from collections import Counter


def plot_image(*images: torch.Tensor, **labeled_images: torch.Tensor) -> None:
    for i, img in enumerate(images):
        labeled_images[f'Image {i+1}'] = img

    fig, axes = plt.subplots(1, len(labeled_images), figsize=(12, 6))
    axes: list[Axes] = [axes] if type(axes) == Axes else axes.flatten()

    for ax, (label, pixels) in zip(axes, labeled_images.items()):
        # HW = [d for d in pixels.shape if d != 3]
        # pixels = pixels.reshape(*HW, 3)
        pixels = pixels.permute(1, 2, 0)
        print(pixels.shape)
        ax.imshow(pixels)
        ax.set_title(label)

    plt.show()
    return


def plot_rgb(*images: np.ndarray, **labeled_images: np.ndarray) -> None:
    for i, img in enumerate(images):
        labeled_images[f'Image {i+1}'] = img

    fig, axes = plt.subplots(1, len(labeled_images), figsize=(12, 6),
                             subplot_kw={'projection': '3d'})
    axes: list[Axes3D] = [axes] if type(axes) == Axes3D else axes.flatten()

    for ax, (label, pixels) in zip(axes, labeled_images.items()):
        pixels = np.asarray(pixels)
        assert 3 in pixels.shape, f'Expected 3 channels, got {pixels.shape}'

        pixels = pixels.reshape(-1, 3)

        color_counter = Counter(zip(pixels[:, 0], pixels[:, 1], pixels[:, 2]))
        R, G, B, freq = zip(*[(r, g, b, c)
                            for (r, g, b), c in color_counter.items()])
        colors = np.array([R, G, B]).T

        ax.scatter(R, G, B, c=colors / 255, alpha=.1, s=freq)
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        ax.set_title(label)
        ax.invert_xaxis()

    plt.show()
    return


if __name__ == '__main__':
    dataset = LOLImageDataset()
    # plot_rgb(**dataset[1])
    plot_image(**dataset[0])
