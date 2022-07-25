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
        self.data_dir=os.path .join(data_dir ,'*.txt')
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
    def __getitem__(self, i):
        ditance_path=self.image_dir [i]
        image_path=ditance_path .replace('distance','image').split('.')[0]+'.tif'
        label_path=image_path .replace('image','label')
        # label_path =label_path .split('.')[0]+'_L.png'
        image=cv2.imread(image_path )
        label=cv2.imread(label_path )
        distance=np.genfromtxt(ditance_path )
        distance =reclass(distance )

        flipcode = random.choice([-1, 0, 1, 2])
        if flipcode != 2:
            image = self.flip(image, flipcode)
            label = self.flip(label, flipcode)
            distance =self .flip(distance ,flipcode )
        else:
            rot_angle = random.choice(range(-10, 10))
            image = self.rotate(image, rot_angle)
            label = self.rotate(label, rot_angle)
            distance =self .rotate(distance ,rot_angle )

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=self.img_transform (image)
        label=cv2.cvtColor(label,cv2.COLOR_BGR2RGB)

        label=self.label_transform.encode_label_img(label)
        label=torch.from_numpy(label)

        return image,label,distance

class Valid_Loader(Dataset ):
    def __init__(self,image_dir):
        self.dir=image_dir
        self.image_dir=glob.glob(os.path .join(self .dir ,'*.png'))

        self.img_transform = torchvision.transforms.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])
            ]
        )
        self.label_transform = LabelProcessor(color_map)
    def __len__(self):
        return len(self.image_dir )
    def __getitem__(self, i):
        img_path=self.image_dir [i]
        label_path=img_path .replace('val_image','val_label')

        image=cv2.imread(img_path )
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB )
        image=self.img_transform(image)

        label=cv2.imread(label_path )
        label=self.label_transform.encode_label_img(label)
        label=torch.from_numpy(label)

        return image,img_path
if __name__=='__main__':
    a=0
    dir_img = r'E:\zxc\WHU\data\train\distance'
    dataset=Data_Loader(dir_img )
    train_loader=torch.utils.data .DataLoader (dataset= dataset ,shuffle= False ,batch_size= 3)
    for image,label,distance in train_loader :
        print(distance.shape)


