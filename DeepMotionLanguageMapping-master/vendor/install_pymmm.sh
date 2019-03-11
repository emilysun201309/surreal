git submodule update --init --recursive
cd vendor/pyMMM/build
cmake ..
make
export PYTHON_PATH=`python -c "import sys; print sys.path[-1]"`
cp _pymmm.so $PYTHON_PATH/.
cp pymmm.py $PYTHON_PATH/.
