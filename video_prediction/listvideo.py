import os

# Getting the current work directory (cwd)
thisdir = os.getcwd()

file_out = open("videolist.txt","w") 

# r=root, d=directories, f = files
for r, d, f in os.walk(thisdir):
    for file in f:
        if ".avi" in file and "m.avi" not in file and "d.avi" not in file:
            file_out.write(os.path.join(r,file) + '\n')
file_out.close() 