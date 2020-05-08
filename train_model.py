import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from model.dncnn import DnCNN
import torchvision.utils as utils
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils import *

parser = argparse.ArgumentParser(description="DnCNN train compression noise for all intra common test sequence")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of tensorboard log files')
parser.add_argument("--saved_path", type=str, default="weights", help='save weight path')
parser.add_argument("--data", type=str, default="data", help='.h5 dataset path')

# qp 42
opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(opt.data, train=True)
    dataset_val   = Dataset(opt.data, train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    model = net.cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            gt, comp = data
            gt_train, comp_train = gt.cuda(), comp.cuda()
            out_train = model(comp_train)
            loss = criterion(out_train, gt_train) / (gt_train.size()[0]*2)
            loss.backward()
            optimizer.step()

            # results
            model.eval()
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, gt_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))

            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            gt = torch.unsqueeze(dataset_val[k][0],  0)
            comp = torch.unsqueeze(dataset_val[k][1],  0)
            with torch.no_grad():
                gt, comp = gt.cuda(), comp.cuda()
                out_val = torch.clamp(model(comp), 0., 1.)
                psnr_val += batch_PSNR(out_val, gt, 1.)
        psnr_val /= len(dataset_val)

        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        with torch.no_grad():
            out_train = torch.clamp(model(comp_train), 0., 1.)

        Img = utils.make_grid(gt_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(comp_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        if epoch % 5 ==0:
            torch.save(model.state_dict(), os.path.join(opt.saved_path, "net_{}.pth".format(epoch)))

if __name__ == "__main__" :
    main()



