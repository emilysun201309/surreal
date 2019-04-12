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
import cv2
from readvideo import parse_seg,readvideo
import math



VIDEO_DATASET = "videos/reshaped.hdf5"
SEG_PATH = "segmentation.txt"
KEYPOINTS = "keypoints.h5"

class NATOPSData(Dataset):
    #__depth = []
    #__flow = []
    #__segm = []
    #__normal = []
    #__video = []
    #__keypoint = []
    #__label = []
    #__appearance = []

    def __init__(self, video_dataset,seg_path,keypoints,data_len=30,transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        self.seg_list = parse_seg(seg_path)
        self.keypoints = pd.HDFStore(keypoints)
        self.motion_video_path = video_dataset
        #self.motion_file = h5py.File(video_dataset,'r+')    
        #file handle for video dataset
        self.data_len = data_len

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        #print('load file')
        '''
        index: 0 - 9599
        subject_no: 0 - 19
        gesture_no: 0 - 23
        numOfRepeat: 0 - 19
        '''
        #subject_idx = int(index/(24*20))
        #gesture_idx = int(index/20%24)
        gesture_idx = int(index/(24*20))
        subject_idx = int(index/20%20)
        numOfRepeat = int(index%20)

        motion = np.zeros((self.data_len,64,64,3),dtype=int)
        keypoint = np.zeros((self.data_len,18))
        
        motion_temp,keypoint_temp = readvideo(self.motion_video_path,subject_idx, gesture_idx, numOfRepeat,self.seg_list,self.keypoints)
        length = len(motion_temp)
        center_x = keypoint_temp[:,2]
        center_x = center_x.reshape((-1,1))
        center_y = keypoint_temp[:,3]
        center_y = center_y.reshape((-1,1))

        keypoint_temp[:,::2] = (keypoint_temp[:,::2] - center_x + 90) * 64/180
        keypoint_temp[:,1:18:2] = (keypoint_temp[:,1:18:2] - center_y + 90) * 64/180

        motion[:min(self.data_len,length)] = motion_temp[:min(self.data_len,length)]
        keypoint[:min(self.data_len,length)] = keypoint_temp[:min(self.data_len,length)]
    
        #print("problem with %d motion data"%index)

        # Convert image and label to torch tensors
        motion = torch.from_numpy(np.asarray(motion))
        #appearance is first frame of motion
        appearance = torch.from_numpy(np.asarray(motion[0]))
        l = np.zeros(24)
        l[gesture_idx] = 1
        l = torch.from_numpy(l)

        keypoint = torch.from_numpy(np.asarray(keypoint))
        ym = {'label':l, 'velocity': keypoint}
        y = (appearance,ym)

        return motion, y

    # Override to give PyTorch size of dataset
    def __len__(self):
        return 24

def main():
    batch_size = 2
    dset_train = NATOPSData(VIDEO_DATASET,SEG_PATH,KEYPOINTS)
    train_loader = DataLoader(dset_train, batch_size, shuffle=True, num_workers=1)
    motion,y = next(iter(train_loader))
    y_a,y_m = y

    print('y_m',y_m)

    #shuffle y_a to get y_a',y_m
    r=torch.randperm(batch_size)
    y_a_prime = y_a[r]
    
    r=torch.randperm(batch_size)
    l_prime = y_m['label'][r]
    k_prime = y_m['velocity'][r]
    y_m_prime = {'label':l_prime,'velocity':k_prime}
    print('y_m_prime',y_m_prime)
    #motion = motion.permute(0,1,4,2,3)
    print("appearance shape", y_a.numpy().shape)
    print("keypoint shape",y_m['velocity'].numpy().shape, "label shape", y_m['label'].numpy().shape)
    print('motion shape:',motion.numpy().shape)
    #frames = int(motion.numpy().shape[1]/10)  
    for i in range(30):
        image = motion.numpy()[0,i,:,:,:]
        print(image.shape)
        plt.imshow(image)
        # plt.show()
        #plt.savefig('render/image%d.png'%i)
        #print(image)
        #plt.close()
        joints_loc = y_m['velocity'].numpy()[0,i,:]
        print('joints shape',joints_loc.shape)
        plt.plot(joints_loc[::2], joints_loc[1:18:2], 'r+')
        #plt.savefig('image%d.png'%f)
        plt.waitforbuttonpress()
        #plt.savefig('render/segm%d.png'%i)
        plt.close()
if __name__ == '__main__':
    main()
