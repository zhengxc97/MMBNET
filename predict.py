import os
import numpy as np
import torch as t
from torch.utils.data import DataLoader
import cv2
from utils.dataloader import Valid_Loader
from model.mmbnet import MMbNet

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

val_dir=r""
res_dir=r""
Load_test = Valid_Loader (val_dir )
test_data = DataLoader(Load_test, batch_size=1, shuffle=False, num_workers=0)

net = MMbNet (3,2)
net.load_state_dict(t.load("./model/best_model.pth"))
net=net.to(device= 'cuda')
net.eval()

colormap =[
    [0,0,0],
    [255,255,255]
]

cm = np.array(colormap).astype('uint8')
if __name__=='__main__':
    for image,dir in test_data :
        valImg = image.to(device='cuda',dtype=t.float)

        out = net(valImg)

        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        pre1=cv2.cvtColor(pre,cv2.COLOR_BGR2RGB)
        out_dir=res_dir+os.path.basename(dir[0])
        cv2.imwrite(out_dir ,pre1)
        print(out_dir)






