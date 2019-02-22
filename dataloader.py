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
        depth= np.zeros((data_len,240,320))
        if(self.__depth[index] != ''):
            depth_temp = np.load(self.__depth[index])
            length = len(depth_temp)
            print(length)
            #if(length <= data_len):
            #    depth = np.pad(depth,((0,data_len-length),(0,0),(0,0)),'constant')
            depth[:min(data_len,length)] = depth_temp[:min(data_len,length)]
            
        flow = np.zeros((data_len,240,320))
        if(self.__flow[index] != ''):
            flow_temp = np.load(self.__flow[index])
            length = len(flow_temp)
            #if(length <= data_len):
            #    flow = np.pad(flow,((0,data_len-length),(0,0),(0,0)),'constant')
            flow[:min(data_len,length)] = flow_temp[:min(data_len,length)]

        segm = np.zeros((data_len,240,320))
        if(self.__segm[index] != ''):
            segm_temp = np.load(self.__segm[index])
            length = len(segm_temp)
            #if(length <= data_len):
            #    segm = np.pad(segm,((0,data_len-length),(0,0),(0,0)),'constant')
            segm[:min(data_len,length)] = segm_temp[:min(data_len,length)]
        
        normal = np.zeros((data_len,240,320))
        if(self.__normal[index] != ''):
            normal_temp = np.load(self.__normal[index])
            length = len(normal_temp)
            #if(length <= data_len):
            #    normal = np.pad(normal,((0,data_len-length),(0,0),(0,0)),'constant')
            normal[:min(data_len,length)] = normal_temp[:min(data_len,length)]

        #annotation
        if(self.__annotation[index] != ''):
            with open(self.__annotation[index]) as f:
                    annotation = json.load(f)
                    print(annotation)

        #images
        img = np.zeros((data_len,240,320,3),dtype=int)
        if(self.__img[index] != ''):
            
            #dimension (H,W,C)
            
            tarfile.open(self.__img[index]).extractall(self.__img[index].split(".")[0])
            
            length = len(glob.glob("%s/*/*" %(self.__img[index].split(".")[0])))
            print('image',length)
            for i in range(min(data_len,length)):
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
        img = torch.from_numpy(np.asarray(img))
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
    
    for i in range((img.numpy()).shape[1]):

        image = img.numpy()[0,i,:,:,:]
        plt.imshow(image)
        #plt.show()
        plt.savefig('render/image%d.png'%i)
        #print(image)
    
        plt.imshow(depth.numpy()[0,i,:,:])
        #plt.show()
        plt.savefig('render/depth%d.png'%i)
        plt.imshow(segm.numpy()[0,i,:,:])
        #plt.show()
        plt.savefig('render/segm%d.png'%i)

if __name__ == '__main__':
    main()
