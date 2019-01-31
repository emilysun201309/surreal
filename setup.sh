#!/bin/bash


#USER_DIR=/home/emily/SURREAL/create_environment
USER_DIR=$1


#create and activate virtual environment
pip install virtualenv
#python3 virtual env
virtualenv -p /usr/bin/python3 venv3
source venv3/bin/activate


#install six
pip install six

#install blender
wget https://download.blender.org/release/Blender2.78/blender-2.78-linux-glibc219-x86_64.tar.bz2
tar xvjf blender-2.78-linux-glibc219-x86_64.tar.bz2
curl https://bootstrap.pypa.io/get-pip.py -o blender-2.78-linux-glibc219-x86_64/2.78/python/lib/python3.5/site-packages/get-pip.py
blender-2.78-linux-glibc219-x86_64/2.78/python/bin/python3.5m blender-2.78-linux-glibc219-x86_64/2.78/python/lib/python3.5/site-packages/get-pip.py
blender-2.78-linux-glibc219-x86_64/2.78/python/bin/python3.5m -m pip install scipy

#OPENEXR
wget https://github.com/openexr/openexr/releases/download/v2.3.0/ilmbase-2.3.0.tar.gz
wget https://github.com/openexr/openexr/releases/download/v2.3.0/openexr-2.3.0.tar.gz
tar xvzf ilmbase-2.3.0.tar.gz
tar xvzf openexr-2.3.0.tar.gz
#install openexr library to USER_DIR/usr/local
mkdir $USER_DIR/usr/local
cd ilmbase-2.3.0
./configure --prefix=$USER_DIR/usr/local
make
make install
cd ../openexr-2.3.0
./configure --prefix=$USER_DIR/usr/local --with-ilmbase-prefix=$USER_DIR/usr/local
cd ..

#exit python 3 environment
deactivate

#python2 virtual env
virtualenv -p /usr/bin/python2.7 venv2
#install python bindings - use python 2
source venv2/bin/activate
pip install numpy
pip install six
easy_install -U openexr
#exit python 2 environment
deactivate

source venv3/bin/activate
#download background files
pip install urllib
cd LSUN
python download.py -c church_outdoor
unzip church_outdoor_train_lmdb.zip

cd ..

#blender's default numpy is problematic
#delete blender's numpy and reinstall
rm -rf blender-2.78-linux-glibc219-x86_64/2.78/python/lib/python3.5/site-packages/numpy
rm -rf blender-2.78-linux-glibc219-x86_64/2.78/python/lib/python3.5/site-packages/numpy-1.16.0.dist-info
blender-2.78-linux-glibc219-x86_64/2.78/python/bin/python3.5m -m pip install numpy


cd datageneration
#change path in config file
python fix_config.py $USER_DIR
./run.sh $USER_DIR





