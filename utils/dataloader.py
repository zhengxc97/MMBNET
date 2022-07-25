import torch
import os
import torch.nn as nn
import torchvision
from PIL import Image
import cv2
from torch.utils .data import Dataset
import torchvision .transforms as T
import glob
import random
import numpy as np

color_map=[
    [0,0,0],
    [255,255,255]
]


def reclass(distance):
    distance[distance <-17]=100
    distance [distance <-13]=200
    distance[distance < -9] = 300
    distance[distance < -4] = 400
    distance[distance < -1 ]= 500
    distance[distance < 1] = 600
    distance[distance < 4] = 700
    distance[distance < 9] = 800
    distance[distance < 13] = 900
    distance [distance <17]=1000
    distance[distance <=20] = 1100
    distance=distance /100
    return distance


class LabelProcessor:
    """对标签图像的编码"""
    def __init__(self, colormap):

        self.colormap = colormap

        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class Data_Loader(Dataset ):
    def __init__(self,data_dir):
        self.data_dir=os.path .join(data_dir ,'*.png')
        self .colormap=color_map
        self.img_transform=torchvision.transforms.Compose(
            [
            T.ToTensor (),
            T.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])
            ]
        )
        self.image_dir=glob.glob(self .data_dir )
        self.label_transform=LabelProcessor (color_map )
    def __len__(self):
        return len(self.image_dir )


    def flip(self, image, flipcode):
        flip = cv2.flip(image, flipcode)
        return flip

    def rotate(self, image, rot_angle):
        rot = cv2.getRotationMatrix2D((256, 256), rot_angle, 1)
        dst = cv2.warpAffine(image, rot, (512, 512))
        return dst


    def RandomCrop(self,image,label,outsize):
        h,w,c=image.shape
        resize=random.choice([h*0.5,h,h*1.5,h*2.0])
        if resize <outsize:
            pad=int(outsize -resize)
            image=cv2.copyMakeBorder(image,pad//2,pad-pad//2,pad//2,pad-pad//2,borderType=cv2.BORDER_DEFAULT)
            label = cv2.copyMakeBorder(label, pad // 2, pad - pad // 2, pad // 2, pad - pad // 2,
                                       borderType=cv2.BORDER_DEFAULT)
        h,w=image.shape[0:2]
        x1=random.randint(0,h-outsize )
        y1=random.randint(0,w-outsize )
        image=image[x1:x1+outsize ,y1:y1+outsize,:]
        label=label[x1:x1+outsize ,y1:y1+outsize ,:]
        return image,label

    def __getitem__(self, i):
        image_path=self.image_dir [i]
        label_path=image_path .replace('image','label')
        # label_path =label_path .split('.')[0]+'_L.png'

        image=cv2.imread(image_path )
        label=cv2.imread(label_path )

        aug_code=random.choice(range(0,100))
        if aug_code<=20:
            rot_angle = random.choice(range(-10, 10))
            image = self.rotate(image, rot_angle)
            label = self.rotate(label, rot_angle)
        elif aug_code<=40 and aug_code >20:
            image = self.flip(image, 0)
            label = self.flip(label, 0)
        elif aug_code <=60 and aug_code >40:
            image = self.flip(image, 1)
            label = self.flip(label, 1)
        else:
            image,label=self.RandomCrop(image,label,512)

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=self.img_transform (image)
        label=cv2.cvtColor(label,cv2.COLOR_BGR2RGB)

        label=self.label_transform.encode_label_img(label)
        label=torch.from_numpy(label)

        return image,label


class Valid_Loader(Dataset ):
    def __init__(self,data_dir):
        self.data_dir=os.path .join(data_dir ,'*.png')
        self .colormap=color_map
        self.img_transform=torchvision.transforms.Compose(
            [
            T.ToTensor (),
            T.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])
            ]
        )
        self.image_dir=glob.glob(self .data_dir )
        self.label_transform=LabelProcessor (color_map )
    def __len__(self):
        return len(self.image_dir )

    def __getitem__(self, i):
        image_path=self.image_dir [i]

        label_path=image_path .replace('val_image','val_label')
        # label_path =label_path .split('.')[0]+'_L.png'
        image=cv2.imread(image_path )
        label=cv2.imread(label_path )

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=self.img_transform (image)
        label=cv2.cvtColor(label,cv2.COLOR_BGR2RGB)

        label=self.label_transform.encode_label_img(label)
        label=torch.from_numpy(label)

        return image,label
if __name__=='__main__':
    a=0
    dir_img = r'E:\zxc\Spacenet\Spacenet\train\image'
    dataset=Data_Loader(dir_img )
    train_loader=torch.utils.data .DataLoader (dataset= dataset ,shuffle= False ,batch_size= 3)
    for image,label in train_loader :
        print(image.shape)


















