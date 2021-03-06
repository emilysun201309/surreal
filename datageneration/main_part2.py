import sys
import os 
from os import remove
from os.path import join, dirname, realpath, exists
import numpy as np
import h5py

def load_body_data(smpl_data, idx=0):
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))
    
    name = sorted(cmu_keys)[idx % len(cmu_keys)]
    
    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data[seq],
                                                   'trans':smpl_data[seq.replace('pose_','trans_')]}
    return(cmu_parms, name)
    
import time
start_time = None
def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))

if __name__ == '__main__':
    # time logging
    #global start_time
    start_time = time.time()
    
    from pickle import load
    import argparse
    
    # parse commandline arguments
    log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--ishape', type=int,
                        help='requested cut, according to the stride')
    parser.add_argument('--stride', type=int,
                        help='stride amount, default 50')

    args = parser.parse_args(sys.argv[sys.argv.index("--idx") :])
    
    idx = args.idx
    ishape = args.ishape
    stride = args.stride
    
    log_message("input idx: %d" % idx)
    log_message("input ishape: %d" % ishape)
    log_message("input stride: %d" % stride)
    
    if idx == None:
        exit(1)
    if ishape == None:
        exit(1)
    if stride == None:
        log_message("WARNING: stride not specified, using default value 50")
        stride = 50
    
    # import idx info (name, split)
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    # get runpass
    #(runpass, idx) = divmod(idx, len(idx_info))
    runpass = idx
    log_message("start part 2")
    
    import hashlib
    import random
    # initialize random seeds with sequence id
    s = "synth_data:%d:%d:%d" % (idx, runpass, ishape)
    seed_number = int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
    log_message("GENERATED SEED %d from string '%s'" % (seed_number, s))
    random.seed(seed_number)
    np.random.seed(seed_number)
    
    # import configuration
    import config
    params = config.load_file('config', 'SYNTH_DATA')
    
    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    resy = params['resy']
    resx = params['resx']
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    output_types = params['output_types']
    stepsize = params['stepsize']
    clipsize = params['clipsize']
    openexr_py2_path = params['openexr_py2_path']
    
    # check whether openexr_py2_path is loaded from configuration file
    if 'openexr_py2_path' in locals() or 'openexr_py2_path' in globals():
        for exr_path in openexr_py2_path.split(':'):
            sys.path.insert(1, exr_path)

    # to read exr imgs
    import OpenEXR 
    import array
    import Imath
    
    log_message("Loading SMPL data")
    #smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))
    #cmu_parms, name = load_body_data(smpl_data, idx)

    #tmp_path = join(tmp_path, 'run%d_%s_c%04d' % (runpass, name.replace(" ", ""), (ishape + 1)))
    tmp_path = join(tmp_path, 'run%d' % (runpass))
    res_paths = {k:join(tmp_path, '%05d_%s'%(idx, k)) for k in output_types if output_types[k]}

    #load data from KIT
    #data = cmu_parms[name]
    poses = np.load(smpl_data_folder + '/KIT/%d.npy'%idx)
    poses = poses[::3]
    trans = np.load(smpl_data_folder+'/KIT/%d_root_pos.npy'%idx)
    trans = trans[::3]

    nframes = len(poses)
    #output_path = join(output_path, 'run%d' % runpass, name.replace(" ", ""))
    output_path = join(output_path, 'run%d' % runpass)
    
    # try loading data using h5py
    #matfile_normal = join(output_path, "%d_c%04d_normal.h5" % (idx,ishape + 1))
    #matfile_gtflow = join(output_path, "%d_c%04d_gtflow.h5" % (idx,ishape + 1))
    #matfile_depth = join(output_path, "%d_c%04d_depth.h5" % (idx,ishape + 1))
    #matfile_segm = join(output_path, "%d_c%04d_segm.h5" % (idx,ishape + 1))
    h5f_data = join(output_path, "%d_c%04d_data.h5" % (idx,ishape + 1))
    normal = np.zeros((nframes,resx,resy,3))
    gtflow = np.zeros((nframes,resx,resy,2))
    depth = np.zeros((nframes,resx,resy))
    segm = np.zeros((nframes,resx,resy))
    get_real_frame = lambda ifr: ifr
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # overlap determined by stride (# subsampled frames to skip)
    #fbegin = ishape*stepsize*stride
    #fend = min(ishape*stepsize*stride + stepsize*clipsize, len(data['poses']))
    # LOOP OVER FRAMES
    for seq_frame, (pose, trans) in enumerate(zip(poses, trans)):
        iframe = seq_frame
        
        log_message("Processing frame %d" % iframe)
        
        for k, folder in res_paths.items():
            if not k== 'vblur' and not k=='fg':
                path = join(folder, 'Image%04d.exr' % get_real_frame(seq_frame))
                exr_file = OpenEXR.InputFile(path)
                if k == 'normal':
                    mat = np.transpose(np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")], (3, resx, resy)), (1, 2, 0))
                    #dict_normal['normal_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False) # +1 for the 1-indexing
                    normal[iframe,:,:] = mat
                elif k == 'gtflow':
                    mat = np.transpose(np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G")], (2, resx, resy)), (1, 2, 0))
                    #dict_gtflow['gtflow_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
                    gtflow[iframe,:,:] = mat
                elif k == 'depth':
                    mat = np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R")], (resx, resy))
                    #dict_depth['depth_%d' % (iframe + 1)] = mat.astype(np.float32, copy=False)
                    depth[iframe,:,:] = mat
                elif k == 'segm':
                    mat = np.reshape([array.array('f', exr_file.channel(Chan, FLOAT)).tolist() for Chan in ("R")], (resx, resy))
                    #dict_segm['segm_%d' % (iframe + 1)] = mat.astype(np.uint8, copy=False)
                    segm[iframe,:,:] = mat
                #remove(path)

    #save as numpy array instead

    '''
    import scipy.io
    scipy.io.savemat(matfile_normal, dict_normal, do_compression=True)
    scipy.io.savemat(matfile_gtflow, dict_gtflow, do_compression=True)
    scipy.io.savemat(matfile_depth, dict_depth, do_compression=True)
    scipy.io.savemat(matfile_segm, dict_segm, do_compression=True)
    '''
    '''
    np.save(matfile_normal,normal)
    np.save(matfile_gtflow,gtflow)
    np.save(matfile_depth,depth)
    np.save(matfile_segm,segm)
    '''
    h5f = h5py.File(h5f_data, 'w')
    h5f.create_dataset('normal', data=normal)
    h5f.create_dataset('gtflow', data=gtflow)
    h5f.create_dataset('depth', data=depth)
    h5f.create_dataset('segm', data=segm)
    h5f.close()
    # cleaning up tmp
    if tmp_path != "" and tmp_path != "/":
        log_message("Cleaning up tmp")
        os.system('rm -rf %s' % tmp_path)


    log_message("Completed batch")