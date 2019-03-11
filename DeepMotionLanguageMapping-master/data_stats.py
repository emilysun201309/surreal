import argparse

import numpy as np
import h5py


def main(args):
    f = h5py.File(args.input, 'r')
    nb_vocabulary = f['vocabulary'].shape[0]
    nb_annotations = f['annotation_inputs'].shape[0]

    # Compute annotation lengths.
    pad_idx = f['vocabulary'][:].tolist().index(f['vocabulary'].attrs['padding_symbol'])
    annotation_lengths = f['annotation_targets'].shape[1] - np.sum(f['annotation_targets'][:] == pad_idx, axis=-1)
    assert len(annotation_lengths) == nb_annotations

    # Compute unique number of motions and their lengths.
    motion_ids = []
    motion_lengths = []
    for motion_idx, _, id_idx in f['mapping']:
        motion_id = f['ids'][id_idx]
        if motion_id in motion_ids:
            continue
        
        motion_ids.append(motion_id)
        motion = f['motion_targets'][id_idx]
        length = motion.shape[0] - np.sum(np.all(motion == 0., axis=-1))
        motion_lengths.append(length)
    nb_motions = len(motion_ids)
    assert len(motion_ids) == len(motion_lengths)
    
    print('Stats:')
    print('  nb_vocabulary:            {}'.format(nb_vocabulary))
    print('  nb_motions:               {}'.format(nb_motions))
    print('  nb_annotations:           {}'.format(nb_annotations))
    print('')
    print('  total motion lengths:     {}'.format(np.sum(motion_lengths)))
    print('  mean motion length:       {} +/- {}'.format(np.mean(motion_lengths), np.std(motion_lengths)))
    print('')
    print('  total annotation lengths: {}'.format(np.sum(annotation_lengths)))
    print('  mean annotation length:   {} +/- {}'.format(np.mean(annotation_lengths), np.std(annotation_lengths)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    main(args)
