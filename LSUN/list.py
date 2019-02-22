import os

# Getting the current work directory (cwd)
thisdir = os.getcwd()

file_out = open("testfile.txt","w") 

# r=root, d=directories, f = files
for r, d, f in os.walk(thisdir):
    for file in f:
        if ".webp" in file:
            file_out.write(os.path.join(r, file) + '\n')
file_out.close() 