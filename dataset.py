import os.path
import h5py
import cv2
import glob
import torch.utils.data as udata
from utils import *


class Dataset(udata.Dataset):
    def __init__(self, path, train=True, h5 = True):
        super(Dataset, self).__init__()
        self.train = train
        self.h5 = h5
        if h5 :
            if self.train:
                self.h5f = h5py.File(os.path.join(path, 'train.h5'), 'r')
            else :
                self.h5f = h5py.File(os.path.join(path, 'test.h5'),'r')
        else :
            self.files_gt = glob.glob(os.path.join(path, "gt", '*.png'))
            self.files_comp = glob.glob(os.path.join(path, "comp", '*.png'))

    def __len__(self):
        if self.h5:
            return len(self.h5f['gt'])
        else :
            return len(self.files_gt)

    def getfilename(self, index):
        return self.files_gt[index]

    def __getitem__(self, index):
        if self.h5:
            gt = np.array(self.h5f['gt'][index])
            comp = np.array(self.h5f['comp'][index])

        else :
            gt = cv2.imread(self.files_gt[index])
            gt = normalized(np.float32(gt[:,:,0]))
            comp = cv2.imread(self.files_comp[index])
            comp = normalized(np.float32(comp[:,:,0]))

        return torch.unsqueeze(torch.Tensor(gt),0), torch.unsqueeze(torch.Tensor(comp),0) # return pair


    def closeH5(self):
        self.h5f.close()


if __name__ =="__main__":
    d = Dataset('test', False, False)
    print(d[0][0].size(), d[0][1].size() )

