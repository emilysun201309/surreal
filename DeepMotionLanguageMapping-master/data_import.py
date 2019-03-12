import argparse
import logging
import os
import json
import xml.etree.cElementTree as ET
import cPickle as pickle
import zipfile
from tempfile import mkdtemp

import numpy as np


def parse_motions(path):
	xml_tree = ET.parse(path)
	xml_root = xml_tree.getroot()
	xml_motions = xml_root.findall('Motion')
	motions = []

	# TODO: currently we only read the first motion, which is usually the movement. Some files also contain other
	# motions, which are usually objects and/or obstacles in the scene. We should be somehow able to handle this better
	# in case the human motion is not the first motion in the file
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
		if name is None:
			raise RuntimeError('<Joint> has no name')
		joint_indexes.append(idx)
		joint_names.append(name)

	frames = []
	xml_frames = xml_motion.find('MotionFrames')
	if xml_frames is None:
		raise RuntimeError('<MotionFrames> not found')
	for xml_frame in xml_frames.findall('MotionFrame'):
		frames.append(_parse_frame(xml_frame, joint_indexes))

	return joint_names, frames


def _parse_frame(xml_frame, joint_indexes):
	n_joints = len(joint_indexes)
	xml_joint_pos = xml_frame.find('JointPosition')
	if xml_joint_pos is None:
		raise RuntimeError('<JointPosition> not found')
	joint_pos = _parse_list(xml_joint_pos, n_joints, joint_indexes)

	return joint_pos


def _parse_list(xml_elem, length, indexes=None):
	if indexes is None:
		indexes = range(length)
	elems = [float(x) for idx, x in enumerate(xml_elem.text.rstrip().split(' ')) if idx in indexes]
	if len(elems) != length:
		raise RuntimeError('invalid number of elements')
	return elems


def main(args):
	input_path = args.input
	output_path = args.output

	# `extractall` is insecure, but a quick way to extract trusted ZIP archives.
	if os.path.isdir(args.input):
		# No need to extract, input is already a folder.
		tmp_path = input_path
	else:
		# Assume ZIP archive as downloaded from dataset website.
		print('Extracting files from "{}" ...'.format(input_path))
		tmp_path = mkdtemp()
		with zipfile.ZipFile(input_path, 'r') as f:
			f.extractall(tmp_path)
		print('done')
		print('')

	print('Scanning files ...')
	files = [f for f in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, f)) and f[0] != '.']
	basenames = list(set([os.path.splitext(f)[0].split('_')[0] for f in files]))
	print('done, {} potential motions and their annotations found'.format(len(basenames)))
	print('')

	# Parse all files.
	print('Processing data in "{}" ...'.format(tmp_path))
	all_ids = []
	all_motions = []
	all_annotations = []
	reference_joint_names = None
	for idx, basename in enumerate(basenames):
		print(basename)
		print('  {}/{} ...'.format(idx + 1, len(basenames))),

		# Load motion.
		mmm_path = os.path.join(tmp_path, basename + '_mmm.xml')
		assert os.path.exists(mmm_path)
		joint_names, frames = parse_motions(mmm_path)[0]
		if reference_joint_names is None:
			reference_joint_names = joint_names[:]
		elif reference_joint_names != joint_names:
			print('skipping, invalid joint_names {}'.format(joint_names))
			continue
		
		# Load annotation.
		annotations_path = os.path.join(tmp_path, basename + '_annotations.json')
		assert os.path.exists(annotations_path)
		with open(annotations_path, 'r') as f:
			annotations = json.load(f)

		# Load metadata.
		meta_path = os.path.join(tmp_path, basename + '_meta.json')
		assert os.path.exists(meta_path)
		with open(meta_path, 'r') as f:
			meta = json.load(f)

		assert len(annotations) == meta['nb_annotations']
		all_ids.append(int(basename))
		all_motions.append(np.array(frames, dtype='float32'))
		all_annotations.append(annotations)
		print('done')
	assert len(all_motions) == len(all_annotations)
	assert len(all_motions) == len(all_ids)
	print('done, successfully processed {} motions and their annotations'.format(len(all_motions)))
	print('')

	# Pickle the results.
	data = {
		'motions': all_motions,
		'annotations': all_annotations,
		'joint_names': reference_joint_names,
		'ids': all_ids,
	}
	print('Pickling dataset ...')
	with open(output_path, 'wb') as f:
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
	print('done')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('input', type=str)
	parser.add_argument('output', type=str)
	main(parser.parse_args())
