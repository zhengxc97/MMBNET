import os
import torch as t
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils.dataloader import Data_Loader,Valid_Loader
import argparse
from utils.metrics import *
from tqdm import tqdm
from model.mmbnet  import  MMbNet
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
device = t.device('cuda') if torch.cuda.is_available() else t.device('cpu')

train_dir=r''

def get_args():
    parser=argparse .ArgumentParser (description='Training network ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e','--epochs', metavar='E', type=int, default=200, help='Number of epochs', dest='epochs')
    parser.add_argument('-b','--batch-size',metavar= 'B',default=4,help='Batch Size',dest='batchsize' )
    parser.add_argument('-lr','--learning-rate',metavar= 'LR',default= 1e-4,help= 'Learning Rate',dest= 'lr')
    parser .add_argument('-n','--num-class',metavar= 'NM',default= 2,help= 'number of class',dest='numclass' )
    return parser.parse_args()

arg=get_args()
Load_train = Data_Loader (train_dir )
train_data = DataLoader(Load_train, batch_size=arg .batchsize, shuffle=True,num_workers= 4)
num_class  = arg .numclass
net =MMbNet(3,2).cuda()
lr=arg.lr
criterion = nn.CrossEntropyLoss().to(device)
optimizer  = optim.Adam(net.parameters(), lr=lr)

def train(model):
    best = [0]
    train_loss = 0
    net = model.train()
    running_metrics_val = runningScore(2)
    for epoch in range(arg .epochs):
        with tqdm(total=len(train_data), desc=f'Epoch {epoch + 1}/{arg .epochs}', unit='img') as pbar:
            running_metrics_val.reset()

            for group in optimizer.param_groups:
                group['lr'] =arg .lr*((1-(epoch/arg .epochs))**0.9)

            for image,label in train_data :

                img_data = image.to(device ='cuda',dtype=t.float32 )
                img_label =label.to(device='cuda',dtype=t .long )

                out = net(img_data)

                loss =criterion(out, img_label)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
                train_loss += loss.item()

                pre_label = out.max(dim=1)[1].data.cpu().numpy()
                true_label = img_label.data.cpu().numpy()
                running_metrics_val.update(true_label, pre_label)

            metrics = running_metrics_val.get_scores()
            for k, v in metrics[0].items():
                print(k, v)
            train_miou = metrics[0]['mIou: ']
            if max(best) <= train_miou:
                best.append(train_miou)
                t.save(net.state_dict(), f'./model/best_model.pth')



if __name__ == "__main__":
    train(net)
