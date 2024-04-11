import argparse
from tqdm import trange

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.dataloading import LOLImageDataset
from model import Cyanet, LossFn


def train(args):
    test_dataset = LOLImageDataset(root=args.dataset,
                                   partition='train')
    test_dataset = LOLImageDataset(root=args.dataset,
                                   partition='test')
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size, shuffle=True)

    if args.resume_from:
        checkpoint = torch.load(args.resume_from)
        model = Cyanet(32)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']

    else:
        model = Cyanet(32)
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          betas=(args.momentum, 0.999),
                          weight_decay=args.weight_decay)

    for epoch in trange(args.epochs):
        for batch in test_loader:
            model.train()
            optimizer.zero_grad()
            gt = batch['gt']
            lq = model(batch['lq'])

            loss_fn = LossFn()
            loss: torch.Tensor = loss_fn(gt=gt, pred=lq)
            loss.backward()

            optimizer.step()

        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer,
                        'args': args}, f'cyanet_{epoch + 1}.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cyanet Training')
    parser.add_argument('--dataset', type=str, metavar='DIR', default='data/LOL/',
                        help='path to dataset')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='B',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume-from', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint-interval', default=10, type=int, metavar='INTERVAL',
                        help='interval between model checkpoints')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
