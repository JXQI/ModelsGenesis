import numpy as np
import os
import glob
from scipy import ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
from os.path import join

class NIIloader(Dataset):
    def __init__(self,path,dataset,transform=None):
        self.path=path
        self.dataset=dataset
        self.transform=transform
        df=pd.read_csv(self.dataset+'.csv')
        self.nii3D=list(df["image"])
        self.labels=list(df["label"])

    def __len__(self):
        return len(self.nii3D)

    def __getitem__(self, item):
        ori_data = sitk.ReadImage(os.path.join(self.path,self.nii3D[item]+'-image.nii.gz'))
        self.image=sitk.GetArrayFromImage(ori_data)
        self.image=self.image[:74]      #TODO:这里只是简单的让程序跑通
        shape=(1,)+self.image.shape
        self.label=int(self.labels[item])
        if self.transform:
            #self.image=self.transform(self.image.reshape(shape))
            self.image = torch.tensor(self.image.reshape(shape))
            self.label=torch.tensor(self.label)
        return self.image,self.label
'''
function: 将测试集无骨折和骨折数据分开
return: 生成train.csv
'''
def train_set():
    df=pd.read_csv("./ribfrac-val-info.csv")
    belign,frac=[],[]
    df=dict(df)
    for i in df.keys():
        df[i]=list(df[i])
    print(df.keys())
    for i in range(len(df['public_id'])-1):
        if df['public_id'][i]!=df['public_id'][i+1] and df['label_id'][i]==0 and df['label_code'][i]==0:
            belign.append(df['public_id'][i])
    for i in set(df['public_id']):
        if i not in belign:
            frac.append(i)
    belign=list(zip(belign,np.zeros(len(belign))))
    frac=list(zip(frac, np.ones(len(frac))))
    belign.extend(frac)
    pd.DataFrame(belign,columns=['image','label']).to_csv('train.csv',index=False)
    print(sorted(belign))
    print(sorted(frac))
    print(len(belign),len(frac))

if __name__=='__main__':
    # #生成train.csv
    # train_set()

    #测试dataloader
    path = '/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-val-images'
    transform = transforms.Compose([transforms.ToTensor()])
    d=NIIloader(path,dataset='train',transform=transform)
    res=[]
    for i in range(len(d)):
        res.append(d[i][0].size()[1])
    print(min(res))


# def showNii(img,step):
#     for i in range(0,img.shape[0],step):
#         plt.imshow(img[i,:,:],cmap='gray')
#         plt.show()

# data_path='/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-train-images/Part1'
# label_path='/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/Part1'
#
# dataname_list=os.listdir(data_path)
# dataname_list=[i for i in dataname_list if i.endswith('.gz')]
# dataname_list.sort()
# #利用SimpleITK的库
# ori_data=sitk.ReadImage(os.path.join(data_path,dataname_list[0]))
# data1=sitk.GetArrayFromImage(ori_data)
# print(data1.shape)
# #showNii(data1,step=10)
# label_list=os.listdir(label_path)
# label_list=[i for i in label_list if i.endswith('.gz')]
# label_list.sort()
# ori_label=sitk.ReadImage(os.path.join(label_path,label_list[0]))
# label1=sitk.GetArrayFromImage(ori_label)
# print(label1.shape)
# showNii(label1,step=10)


