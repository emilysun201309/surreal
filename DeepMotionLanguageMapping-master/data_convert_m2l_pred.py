import argparse
from shutil import copyfile
import cPickle as pickle

import numpy as np
import h5py

from data_process import pad_multidimensional_sequences


def main(args):
    # Make copy of original data.
    print('Copying original dataset ...')
    copyfile(args.dataset, args.output)
    f = h5py.File(args.output, 'r+')
    print('done, annotation_inputs.shape = {}'.format(f['annotation_inputs'].shape))
    print('')

    # Open predictions.
    print('Loading predictions ...')
    with open(args.input, 'rb') as pf:
        predictions = pickle.load(pf)
    print('done, {} predictions loaded'.format(len(predictions['decoded_data'])))
    assert f['annotation_inputs'].shape[0] == len(predictions['decoded_data'])
    print('')

    # Replace motion data.
    print('Extracting replacement motions ...')
    decoded_data = predictions['decoded_data']
    new_motion_inputs = []
    for d in decoded_data:
        ll = d['log_probabilities']
        hypotheses = d['hypotheses']
        assert len(ll) == len(hypotheses)
        sorted_idx = list(reversed(np.argsort(d['log_probabilities'])))[args.hypothesis_idx]
        new_motion_inputs.append(hypotheses[sorted_idx])
    print('done, {}'.format(len(new_motion_inputs)))
    print('')

    # Duplicating data to fill up.
    print('Duplicating data ...')
    finalized_new_motion_inputs = [None for _ in range(f['motion_inputs'].shape[0])]
    for motion_idx, annotation_idx, _ in f['mapping']:
        finalized_new_motion_inputs[motion_idx] = new_motion_inputs[annotation_idx]
    finalized_new_motion_inputs = pad_multidimensional_sequences(finalized_new_motion_inputs, padding='pre')
    assert finalized_new_motion_inputs.shape[0] == f['motion_inputs'].shape[0]
    assert finalized_new_motion_inputs.shape[-1] == f['motion_inputs'].shape[-1]
    print('done, {}'.format(finalized_new_motion_inputs.shape))
    print('')

    print('Finalizing new dataset in "{}" ...'.format(args.output))
    attrs = dict(f['motion_inputs'].attrs)
    del f['motion_inputs']
    f.create_dataset('motion_inputs', data=finalized_new_motion_inputs)
    for k, v in attrs.iteritems():
        f['motion_inputs'].attrs[k] = v
    f.close()
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--hypothesis-idx', type=int, default=0)
    args = parser.parse_args()

    main(args)
