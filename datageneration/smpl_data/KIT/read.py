import argparse
import os
import json
import xml.etree.cElementTree as ET
import logging
from tempfile import TemporaryFile
import numpy as np
import shutil

def parse_motions(path):
    xml_tree = ET.parse(path)
    xml_root = xml_tree.getroot()
    xml_motions = xml_root.findall('Motion')
    motions = []

    if len(xml_motions) > 1:
        logging.warn('more than one <Motion> tag in file "%s", only parsing the first one', path)
    motions.append(_parse_motion(xml_motions[0], path))
    return motions


def _parse_motion(xml_motion, path):
    xml_joint_order = xml_motion.find('JointOrder')
    if xml_joint_order is None:
        raise RuntimeError('<JointOrder> not found')

    joint_names = []
    joint_indexes = []
    for idx, xml_joint in enumerate(xml_joint_order.findall('Joint')):
        name = xml_joint.get('name')
        name,_ = name.split("_")
        if name is None:
            raise RuntimeError('<Joint> has no name')
        elif name:
            joint_indexes.append(idx)
            joint_names.append(name)

    frames = []
    root_pos_frames = []
    root_rot_frames = []
    xml_frames = xml_motion.find('MotionFrames')
    if xml_frames is None:
        raise RuntimeError('<MotionFrames> not found')
    for xml_frame in xml_frames.findall('MotionFrame'):
        frames.append(_parse_frame(xml_frame, joint_indexes)[0])
        root_pos_frames.append(_parse_frame(xml_frame, joint_indexes)[1])
        root_rot_frames.append(_parse_frame(xml_frame, joint_indexes)[2])
    return joint_names, frames,root_pos_frames,root_rot_frames


def _parse_frame(xml_frame, joint_indexes):
    n_joints = len(joint_indexes)
    xml_joint_pos = xml_frame.find('JointPosition')
    xml_root_pos = xml_frame.find('RootPosition')
    xml_root_rot = xml_frame.find('RootRotation')
    if xml_joint_pos is None:
        raise RuntimeError('<JointPosition> not found')
    joint_pos = _parse_list(xml_joint_pos, n_joints, joint_indexes)
    root_pos = _parse_list(xml_root_pos, 3)
    root_rot = _parse_list(xml_root_rot, 3)
    return joint_pos,root_pos,root_rot


def _parse_list(xml_elem, length, indexes=None):
    if indexes is None:
        indexes = range(length)
    elems = [float(x) for idx, x in enumerate(xml_elem.text.rstrip().split(' ')) if idx in indexes]
    if len(elems) != length:
        raise RuntimeError('invalid number of elements')
    return elems


def main():
    input_path = 'samples/'
    
    print('Scanning files ...')
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f[0] != '.']
    basenames = list(set([os.path.splitext(f)[0].split('_')[0] for f in files]))
    print('basenames', basenames)
    print('done, {} potential motions and their annotations found'.format(len(basenames)))
    print('')

    # Parse all files.
    print('Processing data in "{}" ...'.format(input_path))
    all_ids = []
    all_motions = []
    all_annotations = []
    all_metadata = []
    reference_joint_names = None
    for idx, basename in enumerate(basenames):
        print('  {}/{} ...'.format(idx + 1, len(basenames))),

        # Load motion.
        mmm_path = os.path.join(input_path, basename + '_mmm.xml')
        assert os.path.exists(mmm_path)
        joint_names, frames, root_pos_frames,root_rot_frames = parse_motions(mmm_path)[0]
        print(joint_names)
        if reference_joint_names is None:
            reference_joint_names = joint_names[:]
        elif reference_joint_names != joint_names:
            print('skipping, invalid joint_names {}'.format(joint_names))
            continue
        # load annotations
        annotations_path = os.path.join(input_path, basename + '_annotations.json')
        
        os.rename(annotations_path, 'annotations/%s_annotations.json' %idx)
        
        '''
        # Load annotation.
        annotations_path = os.path.join(input_path, basename + '_annotations.json')
        assert os.path.exists(annotations_path)
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Load metadata.
        meta_path = os.path.join(input_path, basename + '_meta.json')
        assert os.path.exists(meta_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        '''
        #assert len(annotations) == meta['nb_annotations']
        all_ids.append(int(basename))
        all_motions.append(np.array(frames, dtype='float32'))
        root_pos_frames = np.array(root_pos_frames,dtype='float32')
        root_rot_frames = np.array(root_rot_frames,dtype='float32')
        np.save('%s_root_pos.npy'%idx,root_pos_frames)
        np.save('%s_root_rot.npy'%idx,root_rot_frames)
        #all_annotations.append(annotations)
        #all_metadata(meta)
        np.save('%s.npy'%idx, np.array(frames, dtype='float32'))
        print('done')
    #assert len(all_motions) == len(all_annotations)
    #assert len(all_motions) == len(all_ids)
    print('done, successfully processed {} motions and their annotations'.format(len(all_motions)))
    print('')

    # At this point, you can do anything you want with the motion and annotation data.
    #all_motions = np.asarray(all_motions)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    main()
