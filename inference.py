import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_gamma
from src.dataloading import LOLImageDataset, YUV2RGB, CYANET_TEST_TF
from src.cyanet import Cyanet, LossFn


yuv2rgb = YUV2RGB()

def post_proc(x: torch.Tensor) -> torch.Tensor:
    return adjust_gamma(yuv2rgb(x), 1)

def test(args):
    device = 'cpu'
    dataset = LOLImageDataset(root=args.dataset,
                              partition='test',
                              transform=CYANET_TEST_TF)
    loader = DataLoader(dataset)

    checkpoint = torch.load(args.checkpoint)
    model = Cyanet()
    model.load_state_dict(checkpoint['state_dict'])

    loss_fn = LossFn().to(device)
    model = model.to(device)

    model.eval()
    for i, batch in enumerate(loader):
        gt = batch['gt']
        lq = batch['lq']
        pred = model(lq)
        save_image(post_proc(lq), f'out/{i}lq.jpg')
        save_image(post_proc(gt), f'out/{i}gt.jpg')
        save_image(post_proc(pred), f'out/{i}pred.jpg')


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cyanet Training')
    parser.add_argument('--dataset', type=str, metavar='DIR', default='data/LOL',
                        help='path to dataset')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='B',
                        help='mini-batch size (default: 64)')
    parser.add_argument('--checkpoint', required=True, type=str, metavar='PATHs',
                        help='path to latest checkpoint (default: none)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    test(args)
