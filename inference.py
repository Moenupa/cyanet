import argparse

import torch
from torchvision.utils import save_image
from src.dataloading import LOLImageDataset
from model import Cyanet, LossFn


def test(args):
    device = 'cpu'
    dataset = LOLImageDataset(root=args.dataset,
                              partition='test')

    checkpoint = torch.load(args.checkpoint)
    model = Cyanet(32, device=device)
    model.load_state_dict(checkpoint['state_dict'])

    loss_fn = LossFn().to(device)

    # model = model.to(device)
    model.eval()
    batch = dataset[0]
    gt = batch['gt']
    lq = batch['lq']
    pred = model(lq.unsqueeze(0)).squeeze()
    save_image(lq, 'lq.jpg')
    save_image(gt, 'gt.jpg')
    save_image(pred, 'pred.jpg')
    

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
