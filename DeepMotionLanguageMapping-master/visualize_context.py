import argparse
import json
import os
from getpass import getpass
import sys
import itertools
import cPickle as pickle

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import h5py
import sqlite3

import Glacier2
import Ice
Ice.loadSlice('-I%s %s' % (Ice.getSliceDir(), os.path.abspath(os.path.join(__file__, '..', 'MotionDatabase.ice'))))
import MotionDatabase


ICE_CLIENT_CONFIG_PATH = os.path.abspath(os.path.join(__file__, '..', 'client.cfg'))


def get_motion(motion_id, db):
    cache_path = '/tmp/pdf_motion_meta_{}.pkl'.format(motion_id)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            motion = pickle.load(f)
        return motion

    motion = db.getMotion(motion_id)
    with open(cache_path, 'wb') as f:
        pickle.dump(motion, f)
    return motion


def get_leave_node(node):
    if len(node.children) == 0:
        return node
    else:
        return [get_leave_node(child) for child in node.children]


def get_node_with_label(label, nodes):
    for node in nodes:
        if node.label == label:
            return node
        if len(node.children) > 0:
            potential_node = get_node_with_label(label, node.children)
            if potential_node is not None:
                return potential_node
    return None


def flatten(l):
    if isinstance(l, list):
        result = []
        for sl in l:
            result += flatten(sl)
        return result
    else:
        return [l]


def main(args):
    # Ask for username and password separately.
    username = raw_input('MotionDB Username: ')
    password = getpass('MotionDB Password: ')
    
    # Configure Ice and Connect to database.
    properties = Ice.createProperties(sys.argv)
    properties.load(ICE_CLIENT_CONFIG_PATH)
    init_data = Ice.InitializationData()
    init_data.properties = properties
    ic = Ice.initialize(init_data)
    router = Glacier2.RouterPrx.checkedCast(ic.getDefaultRouter())
    session = router.createSession(username, password)
    db = MotionDatabase.MotionDatabaseSessionPrx.checkedCast(session)

    mdt = db.getMotionDescriptionTree()
    target_node_ids = set()
    for target_label in args.target_labels:
        target_node = get_node_with_label(target_label, mdt)
        target_node_ids = target_node_ids.union([node.id for node in set(flatten(get_leave_node(target_node)))])
    necessary_node_ids = set()
    for necessary_label in args.necessary_labels:
        necessary_node = get_node_with_label(necessary_label, mdt)
        necessary_node_ids = necessary_node_ids.union([node.id for node in set(flatten(get_leave_node(necessary_node)))])
    print('target_node_ids = {}'.format(target_node_ids))
    print('necessary_node_ids = {}'.format(necessary_node_ids))
    print('')

    f = h5py.File(args.contexts, 'r')
    contexts = f['contexts']
    print('contexts = {}'.format(contexts.shape))
    print('')

    # Figure out mapping.
    print('Computing mapping (for direction {}) ...'.format(args.direction))
    data_f = h5py.File(args.dataset, 'r')
    ids_for_motion_idx = {}
    for motion_idx, annotation_idx, id_idx in data_f['mapping']:
        if args.direction == 'm2l':
            if motion_idx not in ids_for_motion_idx:
                ids_for_motion_idx[motion_idx] = data_f['ids'][id_idx]
            assert ids_for_motion_idx[motion_idx] == data_f['ids'][id_idx]
        else:
            if annotation_idx not in ids_for_motion_idx:
                ids_for_motion_idx[annotation_idx] = data_f['ids'][id_idx]
            assert ids_for_motion_idx[annotation_idx] == data_f['ids'][id_idx]
    assert len(ids_for_motion_idx) == contexts.shape[0]
    ids = [ids_for_motion_idx[idx] for idx in range(contexts.shape[0])]
    assert contexts.shape[0] == len(ids)
    print('done')
    print('')

    # Collect tags from KIT motion database.
    print('Fetching Motion Annotation Tool metadata ...')
    conn = sqlite3.connect(args.database)
    c = conn.cursor()
    motion_db_ids = []
    motion_db_file_ids = []
    for i in ids:
        c.execute('SELECT motion_db_id, motion_db_file_id FROM dataset_motionfile WHERE id=? LIMIT 1', (i,))
        motion_db_id, motion_db_file_id = c.fetchone()
        motion_db_ids.append(motion_db_id)
        motion_db_file_ids.append(motion_db_file_id)
    assert len(motion_db_ids) == len(ids)
    assert len(motion_db_file_ids) == len(ids)
    print('done')
    print('')

    print('Fetching meta information for {} motion entries from KIT motion database ...'.format(len(set(motion_db_ids))))
    labels_for_motion_db_id = {}
    for motion_db_id in set(motion_db_ids):
        nodes = get_motion(motion_db_id, db).motionDescriptions
        label = 'other'

        has_necessary_node = False
        for node in nodes:
            is_necessary_node = (len(necessary_node_ids) == 0 or node.id in necessary_node_ids)
            if is_necessary_node:
                has_necessary_node = True
                break

        if has_necessary_node:
            # Search for the target label.
            for node in nodes:
                is_target_node = node.id in target_node_ids
                if is_target_node:
                    label = node.label
                    break
        labels_for_motion_db_id[motion_db_id] = label
    print('done')
    print('')

    # Filter out duplicate motions.
    print('Filtering out duplicates ...')
    used_motion_file_ids = []
    valid_indexes = []
    for idx, motion_db_file_id in enumerate(motion_db_file_ids):
        if motion_db_file_id in used_motion_file_ids:
            continue
        used_motion_file_ids.append(motion_db_file_id)
        valid_indexes.append(idx)
    valid_contexts = [contexts[idx] for idx in valid_indexes]
    valid_motion_db_ids = [motion_db_ids[idx] for idx in valid_indexes]
    assert len(valid_contexts) == len(valid_motion_db_ids)
    print('done, reduced from {} to {}'.format(len(contexts), len(valid_contexts)))
    print('')

    print('Fitting t-SNE and transforming contexts ...')
    model = TSNE(n_components=2, random_state=0)
    transformed_valid_contexts = model.fit_transform(valid_contexts)
    print('done, valid_transformed_contexts = {}'.format(transformed_valid_contexts.shape))
    print('')

    # Scatter by label.
    clustered_data = {}
    for context, motion_db_id in zip(transformed_valid_contexts, valid_motion_db_ids):
        label = labels_for_motion_db_id[motion_db_id]
        if label not in clustered_data:
            clustered_data[label] = []
        clustered_data[label].append(context)

    # Plot data.
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    print args.colors
    if args.colors is not None:
        colors = itertools.cycle(args.colors)
    else:
        colors = iter(cm.gist_rainbow(np.linspace(0, 1, len(clustered_data))))

    # Draw "other" first so that it does not overlap.
    cs = np.array(clustered_data['other'])
    ax.scatter(cs[:, 0], cs[:, 1], label='other', color='lightgray', marker='o', edgecolors='w', s=17, lw=.3)

    # Draw remaining labels.
    for label, cs in clustered_data.iteritems():
        if label == 'other':
            continue
        color = colors.next()
        cs = np.array(cs)
        ax.scatter(cs[:, 0], cs[:, 1], label=label, color=color, marker='o', edgecolors='w', s=17, lw=.3)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=False, shadow=False, ncol=5)
    if args.output:
        print('Saving plot to "{}" ...'.format(args.output))
        plt.savefig(args.output)
        print('done')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('contexts', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('database', type=str)
    parser.add_argument('direction', choices=['m2l', 'l2m'])
    parser.add_argument('--target-labels', type=str, nargs='+', default=['motion'])
    parser.add_argument('--necessary-labels', type=str, nargs='*', default=[])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--colors', type=str, nargs='*', default=None)
    main(parser.parse_args())
