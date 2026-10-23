import argparse
import os
import time
import shutil
import sys
import traceback
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from models.ST_Former import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_MMEW import train_data_loader, test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=100, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1, type=int, metavar='N')
parser.add_argument('--resume', default=None, type=str, metavar='PATH')
parser.add_argument('--num-classes', default=6, type=int)

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
log_txt_path = './log/' + time_str + 'log.txt'
log_curve_path = './log/' + time_str + 'log.png'
checkpoint_path = './checkpoint/' + time_str + 'model.pth'
best_checkpoint_path = './checkpoint/' + time_str + 'model_best.pth'

os.makedirs('./log', exist_ok=True)
os.makedirs('./checkpoint', exist_ok=True)


def main():
    print("Initializing training script...", flush=True)
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    
    print("Loading model...", flush=True)
    model = GenerateModel(num_classes=args.num_classes)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print("Using GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("WARNING: CUDA not available, using CPU.", flush=True)

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc'].cuda() if torch.cuda.is_available() else checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True
    
    print("Loading datasets...", flush=True)
    train_data = train_data_loader()
    test_data = test_data_loader()

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"Data loaders ready. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch} starting ---", flush=True)
        start_time = time.time()
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)
        val_acc, val_los = validate(val_loader, model, criterion, args)
        scheduler.step()

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict(), 'recorder': recorder}, is_best)

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)
        print(f'Epoch {epoch} completed. Best Accuracy: {best_acc.item():.3f}, Epoch Time: {time.time() - start_time:.1f}s', flush=True)


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses, top1 = AverageMeter('Loss', ':.4f'), AverageMeter('Accuracy', ':6.3f')
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()
        output = model(images)
        loss = criterion(output, target)
        acc1, _ = accuracy(output, target, topk=(1, min(5, args.num_classes)))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}] Loss {losses.val:.4f} ({losses.avg:.4f}) Acc@1 {top1.val:.3f} ({top1.avg:.3f})", flush=True)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses, top1 = AverageMeter('Loss', ':.4f'), AverageMeter('Accuracy', ':6.3f')
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images, target = images.cuda(), target.cuda()
            output = model(images)
            loss = criterion(output, target)
            acc1, _ = accuracy(output, target, topk=(1, min(5, args.num_classes)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
    print(f"Validation Accuracy: {top1.avg:.3f}%", flush=True)
    return top1.avg, losses.avg


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].contiguous().view(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size) for k in topk]


class RecorderMeter:
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.epoch_losses = np.zeros((total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        if isinstance(train_acc, torch.Tensor):
            train_acc = train_acc.item()
        if isinstance(val_acc, torch.Tensor):
            val_acc = val_acc.item()
        self.epoch_losses[idx] = [train_loss * 50, val_loss * 50]
        self.epoch_accuracy[idx] = [train_acc, val_acc]

    def plot_curve(self, save_path):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.epoch_accuracy[:, 0], label='train-accuracy', color='g')
        ax.plot(self.epoch_accuracy[:, 1], label='valid-accuracy', color='y')
        ax.legend()
        fig.savefig(save_path, dpi=80, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)