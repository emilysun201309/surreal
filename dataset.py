#dataset
import os
from os import listdir
import csv


csv_out = open("files.csv", 'w') 
fieldnames = ['depth', 'flow', 'segm','normal','annotation','img']
writer = csv.DictWriter(csv_out,fieldnames=fieldnames)
writer.writeheader()
for subdir, dirs, files in os.walk("out/"):
    for name in dirs:
      print(name)
      directory =  os.path.join("out/", name)
      depth,flow,segm,normal,info,img = None,None,None,None,None,None
      for file in os.listdir(directory):
          if file.endswith("depth.mat"):
            depth = os.path.join(directory, file)
          if file.endswith("gtflow.mat"):
            flow = os.path.join(directory, file)
            
          if file.endswith("segm.mat"):
            segm = os.path.join(directory, file)
            
          if file.endswith("normal.mat"):
            normal = os.path.join(directory, file)
          
          if file.endswith(".json"):
            annotation = os.path.join(directory, file)

          #image folder
          if file.endswith(".mp4.tar.gz"):
            img = os.path.join(directory,file)
            
      writer.writerow({'depth': depth,'flow': flow,'segm': segm,'normal': normal,'annotation': annotation,'img':img})

          
