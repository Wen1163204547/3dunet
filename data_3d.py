import numpy as np
import os
import torch
from torch.utils.data import Dataset

class DataLoader3d(Dataset):
    def __init__(self, ct_path, seg_path, train=True, random=False, black=True, test=False, mask=None):
        self.train = train
        self.test = test
        ct_list = os.listdir(ct_path)
        seg_list = os.listdir(seg_path)
        self.ct_path = [os.path.join(ct_path, i) for i in ct_list if 'volume' in i]
        self.ct_path.sort()
        self.seg_path = [os.path.join(seg_path, i) for i in seg_list if 'segmentation' in i]
        self.seg_path.sort()


    def __getitem__(self, index):
        ct = np.load(self.ct_path[index])
        seg = np.load(self.seg_path[index])
        ct = np.array(ct, dtype=np.float32)
        seg = np.array(seg>0.5, dtype=np.uint8)
        ct = (ct - ct.min()) / (ct.max() - ct.min())
        if self.test:
            return  torch.FloatTensor(ct).unsqueeze(0), torch.LongTensor(seg), self.ct_path[index]
        return torch.FloatTensor(ct).unsqueeze(0), torch.LongTensor(seg)

    def __len__(self):
        return len(self.ct_path)


class TrainLoader2d(Dataset):
    def __init__(self, data_path, list_path, train=True):
        self.train = train
        self.list = np.load(list_path)
        self.data_path = data_path

    def __getitem__(self, idx):
        nid, sid = self.list[idx]
        ct = np.load(os.path.join(self.data_path, 'volume-%d.npy') % nid)[sid-1:sid+2]
        seg = np.load(os.path.join(self.data_path, 'segmentation-%d.npy') % nid)[sid-1:sid+2]
        return torch.FloatTensor(ct), torch.LongTensor(seg)

        
