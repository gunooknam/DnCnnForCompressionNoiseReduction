
import argparse
import glob
import h5py
import numpy as np
import os
import cv2

parser = argparse.ArgumentParser(description="DnCNN train compression noise for all intra common test sequence")
parser.add_argument("--patchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--stride", type=int, default=17, help="Number of total layers")
parser.add_argument("--savepath", type=str, default="data", help='h5 save data path')
parser.add_argument("--datapath", type=str, default="D:\\Test_Sequence\\SelectData_qp_32", help='data path')

opt = parser.parse_args()


def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def scaledPatch(img, h, w , scale, patchSize, stride):
    img = cv2.resize(img, (int(h * scale), int(w * scale)), interpolation=cv2.INTER_CUBIC)
    img = np.expand_dims(img[:,:,0].copy(), 0)
    img = np.float32(normalize(img))
    return Im2Patch(img, win=patchSize, stride=stride)


def prepare_train_test_data(datapath, d, patchSize, stride, savepath):
    print('generate recon.png to cropped h5 data')
    scales = [1, 0.9, 0.8, 0.7] # for augmentation
    diction=dict()
    diction['gt']   = os.path.join( datapath, 'gt', d)
    diction['comp'] = os.path.join( datapath, 'comp',d)

    files_gt = glob.glob(os.path.join(diction['gt'], '*.png'))
    files_comp = glob.glob(os.path.join(diction['comp'], '*.png'))
    n_files = len(files_gt)
    print(n_files)
    files_gt.sort()
    files_comp.sort()
    train_num = 0

    gtList = []
    compList = []
    for i in range(n_files):
        img_ori = cv2.imread(files_gt[i])
        img_comp = cv2.imread(files_comp[i])

        h, w, c = img_ori.shape
        for k in range(len(scales)):
            patches = scaledPatch(img_ori, h, w , scales[k], patchSize, stride)
            patches_comp = scaledPatch(img_comp, h, w , scales[k], patchSize, stride)
            print("file: %s scale %.1f # samples: %d" % (files_gt[i], scales[k], patches.shape[3]))

            for n in range(patches.shape[3]):
                gt   = patches[:,:,:,n].copy()
                comp = patches_comp[:,:,:,n].copy()
                gtList.append(gt)
                compList.append(comp)
                train_num += 1

    # make dataset
    gt = np.vstack(gtList)
    comp = np.vstack(compList)

    # shuffle
    if d == 'train':
        order = np.random.choice(train_num, train_num, replace=False)
        gt =  np.array([gt[i] for i in order])
        comp = np.array([comp[i] for i in order])

    with h5py.File(os.path.join(savepath, d+".h5"), 'w') as hf:
        hf.create_dataset('gt', data=gt)
        hf.create_dataset('comp', data=comp)

if __name__ == '__main__':
    dTrain = ["train", "test"]
    for d in dTrain:
        prepare_train_test_data(opt.datapath, d, 64, 20, opt.savepath)



