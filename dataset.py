#dataset
import os
from os import listdir
import csv


csv_out = open("files.csv", 'w') 
fieldnames = ['data','annotation','img','keypoint']
writer = csv.DictWriter(csv_out,fieldnames=fieldnames)
writer.writeheader()
for subdir, dirs, files in os.walk("out/"):
    for name in dirs:
      print(name)
      directory =  os.path.join("out/", name)
      data,annotation,img = None,None,None
      for file in os.listdir(directory):
          if file.endswith("data.h5"):
            data = os.path.join(directory, file)
          if file.endswith("info.h5")
            keypoint = os.path.join(directory,file)
          '''
          if file.endswith("gtflow.npy"):
            flow = os.path.join(directory, file)
            
          if file.endswith("segm.npy"):
            segm = os.path.join(directory, file)
            
          if file.endswith("normal.npy"):
            normal = os.path.join(directory, file)
          '''
          if file.endswith(".json"):
            annotation = os.path.join(directory, file)

          #image folder
          if file.endswith(".mp4.tar.gz"):
            img = os.path.join(directory,file)
            
      writer.writerow({'data':data,'annotation': annotation,'img':img,'keypoint':keypoint})

          
