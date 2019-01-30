#!/bin/bash

JOB_PARAMS=${1:-'--idx 3 --ishape 0 --stride 50'} # defaults to [2, 0, 50]
#USER_DIR=/home/emily/SURREAL/create_environment
USER_DIR=$1

# SET PATHS HERE
FFMPEG_PATH=/usr/bin/ffmpeg
X264_PATH=/usr/include
PYTHON2_PATH=$USER_DIR/venv2 # PYTHON 2
BLENDER_PATH=$USER_DIR/blender-2.78-linux-glibc219-x86_64
#cd surreal/datageneration

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.78/python
export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.5:${BUNDLED_PYTHON}/lib/python3.5/site-packages
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

# FFMPEG
export LD_LIBRARY_PATH=${FFMPEG_PATH}/lib:${X264_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${FFMPEG_PATH}/bin:${PATH}

for i in {0..11}
do 
    echo "$i"
    JOB_PARAMS="--idx $i --ishape 0 --stride 50"
    ### RUN PART 1  --- Uses python3 because of Blender
    $BLENDER_PATH/blender -b -P main_part1.py -- ${JOB_PARAMS}

    ### RUN PART 2  --- Uses python2 because of OpenEXR
    PYTHONPATH="" ${PYTHON2_PATH}/bin/python2.7 main_part2.py ${JOB_PARAMS}
done
