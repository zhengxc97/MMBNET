import numpy as np
from torch.utils .data import Dataset
from glob import glob
import os
import cv2

color_map=[
  [0,0,0],
[255,255,255]
]


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

class LabelProfessor:
    def __init__(self,colormap):
        self.colormap=colormap
        self.lbl=self.cm2lbl(colormap )

    def cm2lbl(self,colormap):
        lbl=np.zeros(256**3)
        for i,cm in enumerate(colormap ):
            lbl[(cm[0]*256+cm[1])*256+cm[2]]=i
        return lbl

    def encode_label_img(self,img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.lbl[idx], dtype='int64')

class valid_loader(Dataset):
    def __init__(self,dir_label,dir_res):
        self.dir_label=dir_label
        self.label_path=glob(os.path.join(dir_label ,r'*.png'))
        self.la=LabelProfessor(color_map)
        self.res_dir=dir_res
    def __len__(self):
        return len(self.dir_label)
    def __getitem__(self, i):
        label_path=self .label_path [i]
        label=cv2.imread(label_path,0)
        res_path=self.res_dir +os.path.basename(label_path )

        res=cv2.imread(res_path ,0)
        label[label>0]=1
        res[res>0]=1

        return label,res
labelpath=r''
respath=r''
con_matr = np.zeros((2,2))
test_loader = valid_loader(labelpath,respath )
for label ,res in test_loader :
    con_matr +=_fast_hist(label,res,n_class= 2)

pa=np.diag(con_matr )/con_matr .sum(axis= 0)
iou=np.diag(con_matr) / (con_matr.sum(axis=1) + con_matr.sum(axis=0) - np.diag(con_matr))
recall=np.diag(con_matr )/con_matr .sum(axis= 1)
f1=2*pa*recall /(recall +pa)

print('iou:',iou[1])
print('pa:',pa[1])
print('recall:',recall[1])
print('f1:',f1[1])
