from model.dncnn import DnCNN
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from dataset import Dataset
from utils import *

parser = argparse.ArgumentParser(description="DnCNN train compression noise for all intra common test sequence")

parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--modelPath", type=str, default="weights\\net_45.pth", help='save weight path')
parser.add_argument("--resultImgPath", type=str, default="results", help='save weight path')
parser.add_argument("--testImgPath", type=str, default="test", help='.h5 dataset path')

# qp 32
opt = parser.parse_args()

def main():

    print('Loading model ...\n')
    device = torch.device('cuda')
    model = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model.load_state_dict(torch.load(opt.modelPath, map_location=device))
    model.eval()
    dataset_val = Dataset(opt.testImgPath, train=False, h5=False)
    psnr_test = 0

    for k in range(len(dataset_val)):
        gt, comp = dataset_val[k]
        gt = torch.unsqueeze(gt, 0)
        comp = torch.unsqueeze(comp, 0)
        with torch.no_grad():
            out = torch.clamp(model(comp), 0., 1.)
            img = torch.squeeze(torch.squeeze(out,0),0).detach().numpy()
            fname = os.path.basename(dataset_val.getfilename(k))
            img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(os.path.join(opt.resultImgPath, fname), img)

        psnr_before = batch_PSNR(comp, gt, 1.)
        psnr = batch_PSNR(out, gt, 1.)
        psnr_test += psnr
        print("%s PSNR before %f, after %f " % (dataset_val.getfilename(k), psnr_before, psnr))

    psnr_test /= len(dataset_val)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__" :
    main()
