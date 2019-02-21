import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import csv
import json
from PIL import Image
import tarfile
import glob


FOLDER_DATASET = "/out"

class MotionData(Dataset):
    __depth = []
    __flow = []
    __segm = []
    __normal = []
    __annotation = []
    __img = []

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        with open('files.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # depth path
                self.__depth.append(row['depth'])        
                # flow path
                self.__flow.append(row['flow'])
                # segm path
                self.__segm.append(row['segm'])        
                # normal path
                self.__normal.append(row['normal'])
                # annotation path
                self.__annotation.append(row['annotation'])
                #image path
                self.__img.append(row['img'])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        data_len = 130
        depth= np.nan
        if(self.__depth[index] != ''):
            print(self.__depth[index])
            depth = loadmat(self.__depth[index])
            depth = [ v for v in depth.values()]
            
            if(len(depth)>3):
                depth = depth[3:]
                np.stack(depth, axis=0)
                depth = np.stack(depth, axis=0)
                length = len(depth)
                if(length <= data_len):
                    depth = np.pad(depth,((0,data_len-length),(0,0),(0,0)),'constant')
            else:
                depth = np.nan
        flow = np.nan
        if(self.__flow[index] != ''):
            flow = loadmat(self.__flow[index])
            print(self.__flow[index])
            flow = [ v for v in flow.values()]
            
            if(len(flow)>3):
                flow = flow[3:]
                np.stack(flow, axis=0)
                flow = np.stack(flow, axis=0)
                length = len(flow)
                if(length <= data_len):
                    flow = np.pad(flow,((0,data_len-length),(0,0),(0,0)),'constant')
            else:
                flow = np.nan
        segm = np.nan
        if(self.__segm[index] != ''):
            segm = loadmat(self.__segm[index])
            print(self.__segm[index])
            segm = [ v for v in segm.values()]
            
            if(len(segm)>3):
                segm = segm[3:]
                np.stack(segm, axis=0)
                segm = np.stack(segm, axis=0)
                length = len(segm)
                if(length <= data_len):
                    segm = np.pad(segm,((0,data_len-length),(0,0),(0,0)),'constant')
            else:
                segm = np.nan
        normal = np.nan
        if(self.__normal[index] != ''):
            normal = loadmat(self.__normal[index])
            print(self.__normal[index])
            normal = [ v for v in normal.values()]
            
            if(len(normal)>3):
                normal = normal[3:]
                np.stack(normal, axis=0)
                normal = np.stack(normal, axis=0)
                length = len(normal)
                if(length <= data_len):
                    normal = np.pad(normal,((0,data_len-length),(0,0),(0,0)),'constant')
            else:
                normal = np.nan

        #annotation
        if(self.__annotation[index] != ''):
            with open(self.__annotation[index]) as f:
                    annotation = json.load(f)
                    print(annotation)

        #images
        if(self.__img[index] != ''):
            img = np.zeros((data_len,240,320,3),dtype=int)
            #dimension (H,W,C)
            
            tarfile.open(self.__img[index]).extractall(self.__img[index].split(".")[0])
            
            for i in range(data_len):
                path = glob.glob("%s/*/Image%04d.png" %(self.__img[index].split(".")[0],i))
                print('path',path)
                with Image.open(path[0]) as img_temp:
                    img_temp = img_temp.convert('RGB')
                    img_temp = np.asarray(img_temp)
                    img_temp = img_temp.astype(int)
                    img[i] = img_temp
                #img[i] = torch.from_numpy(img_temp,dtype=torch.int)
                
            
        # Convert image and label to torch tensors
        depth = torch.from_numpy(np.asarray(depth))
        
        print('depth',depth.shape)
        flow = torch.from_numpy(np.asarray(flow))
        print('flow',flow.shape)
        segm = torch.from_numpy(np.asarray(segm))
        print('segm',segm.shape)
        normal = torch.from_numpy(np.asarray(normal))
        print('normal',normal.shape)
        normal = torch.from_numpy(np.asarray(img))
        print('image', img.shape)
        return depth,flow,segm,normal,annotation,img

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__depth)

def main():
    dset_train = MotionData(FOLDER_DATASET)
    train_loader = DataLoader(dset_train, batch_size=5, shuffle=True, num_workers=1)
    depth,flow,segm,normal,annotation,img = next(iter(train_loader))
    print('Batch shape:',depth.numpy().shape, flow.numpy().shape,img.numpy().shape)
    image = img.numpy()[0,0,:,:,:]
    
    #print(image)
    plt.imshow(image)
    plt.show()
    plt.imshow(depth.numpy()[0,0,:,:])
    plt.show()
    plt.imshow(segm.numpy()[0,0,:,:])
    plt.show()

if __name__ == '__main__':
    main()
