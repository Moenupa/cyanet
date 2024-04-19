import argparse
import os
from datetime import datetime
from tqdm import trange
import wandb

import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from src.dataloading import LOLImageDataset, CYANET_TRAIN_TF, CYANET_TEST_TF
from src.cyanet import Cyanet, LossFn


def train(args):
    os.makedirs(f'{args.ckpt_root}/{args.exp}')
    wandb.init(project='cyanet', name=args.exp, config=args)
    wandb.save(f"src/cyanet/cyanet.py")

    train_dataset = LOLImageDataset(root=args.dataset,
                                    partition='train',
                                    transform=CYANET_TRAIN_TF)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_dataset = LOLImageDataset(root=args.dataset,
                                   partition='test',
                                   transform=CYANET_TEST_TF)
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
            loss, l_pixel, l_psnr, l_ssim, l_lpips = loss_fn(
                gt=gt,
                pred=pred
            )
            loss.backward()
            wandb.log({'training loss': loss.item(),
                       'training pixel loss': l_pixel.item(),
                       'training pnsr loss': l_psnr.item(),
                       'training ssim loss': l_ssim.item(),
                       'training lpips loss': l_lpips.item()})

            optimizer.step()

        # update learning rate after each epoch, and save model checkpoint
        wandb.log({'lr': scheduler.get_last_lr()[0]})
        scheduler.step()
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
                    loss, l_pixel, l_psnr, l_ssim, l_lpips = loss_fn(
                        gt=gt,
                        pred=pred
                    )
                    wandb.log({'test loss': loss.item(),
                               'test pixel loss': l_pixel.item(),
                               'test pnsr loss': l_psnr.item(),
                               'test ssim loss': l_ssim.item(),
                               'test lpips loss': l_lpips.item()})


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cyanet Training')
    parser.add_argument('--exp', type=str, metavar='NAME', 
                        default=f'{datetime.now():%y%m%d-%H%M}', 
                        help='name of experiment (default: YYmmdd-HHMM)')
    parser.add_argument('--dataset', type=str, metavar='DIR', 
                        default='data/LOL',
                        help='path to dataset (default: data/LOL)')
    parser.add_argument('--device', type=str, metavar='DEVICE',
                        default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='device to use (default: cuda:0)')

    parser.add_argument('--epochs', type=int, metavar='N',
                        default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('-b', '--batch-size', type=int, metavar='B', 
                        default=64, 
                        help='mini-batch size (default: 64)')

    parser.add_argument('--lr', '--learning-rate', type=float, metavar='LR', 
                        default=2e-4, 
                        help='initial learning rate (default: 2e-4)')
    parser.add_argument('--momentum', type=float, metavar='M',
                        default=0.9, 
                        help='momentum for optimizer (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', type=float, metavar='WD', 
                        default=1e-4, 
                        help='weight decay (default: 1e-4)')

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
