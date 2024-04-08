import argparse
from torch.utils.data import dataloader
from src.dataloading import LOLImageDataset, DEFAULT_YUV_TRANSFORM
from model import Cyanet


def train(args):
    test_dataset = LOLImageDataset(transform=DEFAULT_YUV_TRANSFORM)
    test_loader = dataloader(test_dataset)
    
    optimizer = None
    
    model = Cyanet(32)
    
    for batch in test_loader:
        model.train()
        pass
        
    print(model)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cyanet Training')
    parser.add_argument('--data', type=str, metavar='DIR',
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
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
