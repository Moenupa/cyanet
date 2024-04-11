import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from dataloading import LOLImageDataset, RGB255_TO_YUV_TRANSFORM, YUV2RGB, RGB2YUV
from collections import Counter


def plot_image(*images: torch.Tensor, **labeled_images: torch.Tensor):
    for i, img in enumerate(images):
        labeled_images[f'Image {i+1}'] = img

    rows, cols = len(labeled_images) // 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows))
    axes: list[Axes] = [axes] if type(axes) == Axes else axes.flatten()

    for ax, (label, pixels) in zip(axes, labeled_images.items()):
        pixels = pixels.permute(1, 2, 0)
        ax.imshow(pixels)
        ax.set_title(label)

    fig.tight_layout()
    fig.savefig('./res/img.png')
    return fig


def plot_rgb(*images: torch.Tensor, **labeled_images: torch.Tensor):
    for i, img in enumerate(images):
        labeled_images[f'Image {i+1}'] = img

    rows, cols = len(labeled_images) // 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows),
                             subplot_kw={'projection': '3d'})
    axes: list[Axes3D] = [axes] if type(axes) == Axes3D else axes.flatten()

    for ax, (label, pixels) in zip(axes, labeled_images.items()):
        assert 3 in pixels.shape, f'Expected 3 channels, got {pixels.shape}'
        pixels = pixels.reshape(-1, 3).numpy()

        color_counter = Counter(zip(pixels[:, 0], pixels[:, 1], pixels[:, 2]))
        R, G, B, freq = zip(*[(r, g, b, c)
                            for (r, g, b), c in color_counter.items()])
        colors = torch.Tensor([R, G, B]).T
        ax.scatter(R, G, B, c=colors / 255, alpha=.1, s=freq)
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_zlim(0, 255)
        ax.set_title(label)
        ax.invert_xaxis()

    for ax in axes[1:]:
        ax.shareview(axes[0])

    fig.tight_layout()
    fig.savefig('./res/rgb.png')
    return fig


def plot_yuv(*images: torch.Tensor, **labeled_images: torch.Tensor):
    for i, img in enumerate(images):
        labeled_images[f'Image {i+1}'] = img

    rows, cols = len(labeled_images) // 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows),
                             subplot_kw={'projection': '3d'})
    axes: list[Axes3D] = [axes] if type(axes) == Axes3D else axes.flatten()

    for ax, (label, pixels) in zip(axes, labeled_images.items()):
        assert 3 in pixels.shape, f'Expected 3 channels, got {pixels.shape}'
        # (n, 3) in YUV space
        yuv = RGB255_TO_YUV_TRANSFORM(pixels).flatten(1).numpy()

        counter = Counter(zip(yuv[0, :], yuv[1, :], yuv[2, :]))
        n_colors = len(counter)
        Y, U, V, freq = zip(*[(y, u, v, c)
                            for (y, u, v), c in counter.items()])
        colors = YUV2RGB.yuv2rgb(
            torch.Tensor([Y, U, V]).reshape(3, -1, 1)
        ).flatten(1).permute(1, 0).numpy()
        assert colors.shape == (n_colors, 3), f'{colors.shape} {n_colors}'
        # assert not colors.shape, f'{colors.shape} {colors.min()} {colors.max()}'

        ax.scatter(U, V, Y, c=colors, alpha=.1, s=freq)
        ax.set_xlabel('U')
        ax.set_ylabel('V')
        ax.set_zlabel('Y')
        ax.set_xlim(-0.436, 0.436)
        ax.set_ylim(-0.615, 0.615)
        ax.set_zlim(0, 1)
        ax.set_title(label)
        ax.invert_yaxis()

    for ax in axes[1:]:
        ax.shareview(axes[0])

    fig.tight_layout()
    fig.savefig('./res/yuv.png')
    return fig


if __name__ == '__main__':
    dataset = LOLImageDataset(root='data/LOL')
    plot_rgb(**dataset[0])
    plot_yuv(**dataset[0])
    plot_image(**dataset[0])
    plt.show()
