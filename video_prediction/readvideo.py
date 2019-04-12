import cv2
import numpy as np
import re
import h5py
#parse txt file
#frame_list = [[[]]*20]*24
#TODO: LIST comprehension 
#VIDEO_FILE = "videos/video1.hdf5"
def parse_seg(segmentation_path):
    seg_list = [[ [] for _ in range(20)] for _ in range(24)]
    motion_list = []
    with open(segmentation_path, 'r') as f:
        for line in f:
            #print('line',line)
            if '<eof>' in line:
                return seg_list
            if not line.strip():
                continue
            else: 
                line = line.strip()
                if '//' in line:
                    continue
                if ':' not in line:
                    subject_no,gesture_no,numOfRepeat = line.split(',')
                    #print("subject no %s gesture %s "%(subject_no,gesture_no))
                else:
                    #print("split line",re.split('[: ,]',line))
                    no,start,end = re.split('[: ,]',line)
                    seg_list[int(gesture_no)-1][int(subject_no)-1].append((int(start),int(end)))


def readvideo(video_dataset_path,subject_idx, gesture_idx, numOfRepeat,seg_list,store):
    '''
    subject_no,gesture_no for the video [0:19] [0:24]
    numOfRepeat: index of the repeat 
    seg_list: list contain range of frames
    store: handle for .hdf5 file storing the keypoints
    '''
    f = h5py.File(video_dataset_path,'r')
    #print("open file")
    file_name = "g%02ds%02d"%(gesture_idx+1,subject_idx+1)
    image_arr = f[file_name]
    frame_range = seg_list[gesture_idx][subject_idx][numOfRepeat]
    motion = image_arr[frame_range[0]:frame_range[1]]
    keypoints = store['g%02ds%02d'%(gesture_idx+1,subject_idx+1)].iloc[frame_range[0]+1:frame_range[1]+1]
    keypoints = keypoints.values

    return motion,keypoints
    
    

seg_list = parse_seg("segmentation.txt")
print(seg_list[0])
