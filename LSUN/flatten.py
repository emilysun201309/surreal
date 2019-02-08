import shutil
import os


PATH = /home/emily/SURREAL/surreal/data/LSUN/church_outdoor_tr_img/

def move(destination, depth=None):
    if not depth:
        depth = []
    for file_or_dir in os.listdir(os.path.join([destination] + depth, os.sep)):
        if os.path.isfile(file_or_dir):
            shutil.move(file_or_dir, destination)
        else:
            move(destination, os.path.join(depth + [file_or_dir], os.sep))

move(os.path.abspath('/home/emily/SURREAL/surreal/data/LSUN/church_outdoor_tr_img/'), depth=None)
