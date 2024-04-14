import argparse
from tqdm import trange

import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from src.dataloading import LOLImageDataset
from src.cyanet import Cyanet, LossFn


def train(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset = LOLImageDataset(root=args.dataset,
                              partition='train')
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True)

    if args.resume_from:
        checkpoint = torch.load(args.resume_from)
        model = Cyanet(32, device=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

    else:
        model = Cyanet(32, device=device)
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          betas=(args.momentum, 0.999),
                          weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 200, eta_min=1e-6)
    loss_fn = LossFn().to(device)

    model = model.to(device)
    for epoch in trange(args.epochs):
        model.train()
        for batch in loader:
            gt = batch['gt'].to(device)
            lq = batch['lq'].to(device)
            optimizer.zero_grad()

            loss: torch.Tensor = loss_fn(
                gt=gt,
                pred=model(lq)
            )
            loss.backward()

            optimizer.step()

        # update learning rate after each epoch, and save model checkpoint
        scheduler.step()
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer,
                'scheduler': scheduler,
                'args': args
            }, f'{args.model_path}cyanet_{epoch + 1}.pth')

        if args.evaluate:
            # model.eval()
            pass


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cyanet Training')
    parser.add_argument('--dataset', type=str, metavar='DIR', default='data/LOL',
                        help='path to dataset')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='B',
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 2e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume-from', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model-path', default='model/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: model/)')
    parser.add_argument('--checkpoint-interval', default=10, type=int, metavar='INTERVAL',
                        help='interval between model checkpoints')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
