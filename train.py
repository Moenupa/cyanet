import argparse
import os
from datetime import datetime
from tqdm import trange
import wandb

import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from src.dataloading import LOLImageDataset
from src.cyanet import Cyanet, LossFn


def train(args):
    os.makedirs(f'{args.ckpt_root}/{args.exp}')
    wandb.init(project='cyanet', name=args.exp, config=args)
    wandb.save(f"src/cyanet/cyanet.py")

    train_dataset = LOLImageDataset(root=args.dataset,
                                    partition='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_dataset = LOLImageDataset(root=args.dataset,
                                   partition='test')
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size)

    if args.resume_from:
        checkpoint = torch.load(args.resume_from)
        model = Cyanet()
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        start_from_epoch = checkpoint['epoch'] + 1
    else:
        model = Cyanet()
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          betas=(args.momentum, 0.999),
                          weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 200, eta_min=1e-6
        )
        start_from_epoch = 0

    loss_fn = LossFn().to(args.device)
    model = model.to(args.device)

    for epoch in trange(start_from_epoch, args.epochs):
        model.train()
        for batch in train_loader:
            gt = batch['gt'].to(args.device)
            lq = batch['lq'].to(args.device)
            optimizer.zero_grad()
            pred = model(lq)

            loss: torch.Tensor = loss_fn(
                gt=gt,
                pred=pred,
            )
            loss.backward()
            wandb.log({'training loss': loss.item()})

            optimizer.step()

        # update learning rate after each epoch, and save model checkpoint
        scheduler.step()
        wandb.log({'lr': scheduler.get_last_lr()[0]})
        if (epoch + 1) % args.ckpt_interval == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer,
                'scheduler': scheduler,
                'epoch': epoch,
                'args': args
            }, f'{args.ckpt_root}/{args.exp}/cyanet_{epoch + 1}.pth')

        if args.eval and (epoch + 1) % args.eval_interval == 0:
            model.eval()
            for batch in test_loader:
                gt = batch['gt'].to(args.device)
                lq = batch['lq'].to(args.device)

                with torch.no_grad():
                    pred = model(lq)
                    loss: torch.Tensor = loss_fn(
                        gt=gt,
                        pred=pred
                    )
                    wandb.log({'test loss': loss.item()})


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cyanet Training')
    parser.add_argument('--exp', default=f'{datetime.now():%y%m%d-%H%M}', type=str,
                        metavar='EXP', help='name of experiment (default: YYmmdd-HHMM)')
    parser.add_argument('--dataset', type=str, metavar='DIR', default='data/LOL',
                        help='path to dataset (default: data/LOL)')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        type=str, metavar='DEVICE', help='device to use (default: cuda:0)')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='B', help='mini-batch size (default: 64)')

    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 2e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for optimizer (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # checkpointing
    parser.add_argument('--resume-from', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--ckpt-root', default='model', type=str, metavar='PATH',
                        help='root dir to save checkpoint (default: model)')
    parser.add_argument('--ckpt-interval', default=10, type=int, metavar='INTERVAL',
                        help='interval between saving checkpoints (default: 10)')

    # evaluation
    parser.add_argument('-e', '--eval', dest='eval', action='store_true',
                        help='evaluate model on test set (default: False)')
    parser.add_argument('--eval-interval', default=10, type=int, metavar='INTERVAL',
                        help='interval between evaluating model (default: 10)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)
