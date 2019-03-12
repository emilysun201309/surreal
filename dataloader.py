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
import h5py


FOLDER_DATASET = "/out"

class MotionData(Dataset):
    #__depth = []
    #__flow = []
    #__segm = []
    #__normal = []
    __data = []
    __annotation = []
    __img = []
    __keypoint = []

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        with open('files.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # depth path
                #self.__depth.append(row['depth'])        
                # flow path
                #self.__flow.append(row['flow'])
                # segm path
                #self.__segm.append(row['segm'])        
                # normal path
                #self.__normal.append(row['normal'])
                self.__data.append(row['data'])
                # annotation path
                self.__annotation.append(row['annotation'])
                #image path
                self.__img.append(row['img'])
                self.__keypoint.append(row['keypoint'])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        print('load file')
        h5f = h5py.File(self.__data[index],'r')
        print('finish loading')
        data_len = 130
        depth= np.zeros((data_len,240,320))
        try:
            print('load depth')
            depth_temp = h5f['depth'][:] 
            print('finish loading')
            length = len(depth_temp)
            #if(length <= data_len):
            #    depth = np.pad(depth,((0,data_len-length),(0,0),(0,0)),'constant')
            depth[:min(data_len,length)] = depth_temp[:min(data_len,length)]
        except:
            print("problem with %d depth data"%index)
              
        flow = np.zeros((data_len,240,320,2))
        try:
            flow_temp = h5f['gtflow'][:] 
            length = len(flow_temp)
            #if(length <= data_len):
            #    flow = np.pad(flow,((0,data_len-length),(0,0),(0,0)),'constant')
            flow[:min(data_len,length)] = flow_temp[:min(data_len,length)]
        except:
            print("problem with %d flow data"%index)
        segm = np.zeros((data_len,240,320))
        try:
            segm_temp = h5f['segm'][:] 
            length = len(segm_temp)
            #if(length <= data_len):
            #    segm = np.pad(segm,((0,data_len-length),(0,0),(0,0)),'constant')
            segm[:min(data_len,length)] = segm_temp[:min(data_len,length)]
        except:
            print("problem with %d segm data"%index)
        normal = np.zeros((data_len,240,320,3))
        try:
            normal_temp = h5f['normal'][:] 
            length = len(normal_temp)
            #if(length <= data_len):
            #    normal = np.pad(normal,((0,data_len-length),(0,0),(0,0)),'constant')
            normal[:min(data_len,length)] = normal_temp[:min(data_len,length)]
        except:
            print("problem with %d normal data"%index)
        #annotation
        try:
            with open(self.__annotation[index]) as f:
                    annotation = json.load(f)
        except:
            print("problem with annotation")
        #images
        img = np.zeros((data_len,240,320,3),dtype=int)
        try:
            
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
        except:
            print("problem with %d image"%index)        

        #keypoint
        h5f_keypoint = h5py.File(self.__keypoint[index],'r')

        keypoint = np.zeros((2,43,data_len))
        try:
            keypoint_temp = h5f_keypoint['joints2D'][:] 
            length = len(keypoint_temp.shape[2])
            #if(length <= data_len):
            #    normal = np.pad(normal,((0,data_len-length),(0,0),(0,0)),'constant')
            keypoint[:,:,:min(data_len,length)] = keypoint_temp[:,:,:min(data_len,length)]
        except:
            print("problem with %d keypoint data"%index)
   
        # Convert image and label to torch tensors
        depth = torch.from_numpy(np.asarray(depth))
        
        
        flow = torch.from_numpy(np.asarray(flow))
        
        segm = torch.from_numpy(np.asarray(segm))
        
        normal = torch.from_numpy(np.asarray(normal))
        
        img = torch.from_numpy(np.asarray(img))

        keypoint = torch.from_numpy(np.asarray(keypoint))
        
        return depth,flow,segm,normal,annotation,img,keypoint

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__data)

def main():
    dset_train = MotionData(FOLDER_DATASET)
    train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=1)
    depth,flow,segm,normal,annotation,img,keypoint = next(iter(train_loader))
    print('Batch shape:',depth.numpy().shape, flow.numpy().shape,img.numpy().shape)
    frames = int(img.numpy().shape[1]/10)  
    for i in range(frames):
        idx = i * 10
        image = img.numpy()[0,idx,:,:,:]
        
        ax0 = plt.subplot(231)
        ax0.imshow(image)
        # plt.show()
        #plt.savefig('render/image%d.png'%i)
        #print(image)
        #plt.close()
        joints_loc = keypoint[:,:,i]
        ax0.plot(joints_loc[0,:], 240-joints_loc[1,:], 'r+')
        #plt.savefig('image%d.png'%f)
        #plt.close()


        ax1 = plt.subplot(232)
        ax1.imshow(depth.numpy()[0,idx,:,:])
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.savefig('render/depth%d.png'%i)
        ax2 = plt.subplot(233)
        ax2.imshow(segm.numpy()[0,idx,:,:])
        #plt.show()
        ax3 = plt.subplot(234)
        ax3.imshow(flow.numpy()[0,idx,:,:,0])
        ax4 = plt.subplot(235)
        ax4.imshow(normal.numpy()[0,idx,:,:,:])
        plt.waitforbuttonpress()
        #plt.savefig('render/segm%d.png'%i)
        #plt.close()
if __name__ == '__main__':
    main()
