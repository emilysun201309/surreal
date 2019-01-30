import sys
USER_DIR = str(sys.argv[1])
# Read in the file
with open('config', 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace('$USER_DIR', USER_DIR)

# Write the file out again
with open('config', 'w') as file:
  file.write(filedata)