# coding=utf8
import argparse
import os
import cPickle as pickle
import re
import json
from copy import deepcopy

import enchant
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences


end_value = 0.
start_value = 1.


# Adopted from keras to support multi-dimensional time series.
def pad_multidimensional_sequences(sequences, maxlen=None, dtype='float32', padding='pre', padding_value=0., truncating='pre'):
    lengths = [len(s) for s in sequences]
    dims = sequences[0].shape[1:]
    
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen) + dims) * padding_value).astype(dtype)
    for idx, s in enumerate(sequences):
        assert s.shape[1:] == dims
        if len(s) == 0:
            continue # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        
        if padding == 'post':
            x[idx, :len(trunc), :] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):, :] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def normalize_sequences(sequences, factor=1):
    normalized_sequences = []
    for seq in sequences:
        assert seq.ndim == 2
        l = len(seq)
        if l % factor != 0:
            nb_missing_values = factor - (l % factor)
            missing_values = np.repeat(seq[-1, :].reshape((1, seq.shape[1])), nb_missing_values, axis=0)
            assert missing_values.shape == (nb_missing_values, seq.shape[1])
            seq = np.vstack([seq, missing_values])
        assert seq.shape[0] % factor == 0
        normalized_sequences.append(seq)
    return normalized_sequences


def downsample_sequences(sequences, lengths, factor=1):
    assert sequences.ndim == 3

    # Downsample for all possible offsets. We do this since we don't want to
    # waste data.
    updated_sequences = []
    for offset in xrange(factor):
        updated_sequences.append(sequences[:, offset::factor, :])
    updated_sequences = np.concatenate(updated_sequences)

    # Update lengths.
    updated_lengths = []
    for offset in xrange(factor):
        updated_lengths.append((np.ceil(lengths / float(factor))).astype(int))
    updated_lengths = np.concatenate(updated_lengths)

    return updated_sequences, updated_lengths


def tokenize(sent):
    numbers = {
        'ninety': '90',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'zero': '0',
    }
    tokens = sent.split()
    for idx, token in enumerate(tokens):
        if token in numbers:
            tokens[idx] = numbers[token]

    # Split all numbers.
    for idx, token in enumerate(tokens):
        tokens[idx] = [subtoken for subtoken in re.split('(\d)', token) if subtoken]
    tokens = sum(tokens, [])
    return tokens


def process_motions(motions, maxlen, args):
    print('Normalizing and padding motions ...')
    motions = normalize_sequences(motions, factor=args.motion_downsample_factor)
    motion_lengths = np.array([len(m) for m in motions])
    assert np.allclose(motion_lengths % args.motion_downsample_factor, 0)

    # Create a copy of the original motion which we'll use for the target.
    motion_targets = [np.copy(m) for m in motions]
    motion_targets = pad_multidimensional_sequences(motion_targets, maxlen=maxlen, padding='post',
        padding_value=args.motion_padding_value)

    # Prepare the input motions.
    motion_inputs = pad_multidimensional_sequences(motions, maxlen=maxlen, padding='pre',
        padding_value=args.motion_padding_value)
    print('done, motion inputs = {}, motion targets = {}'.format(motion_inputs.shape, motion_targets.shape))
    print('')

    # Down-sample the motions, if applicable.
    motion_input_lengths = motion_lengths[:]
    motion_target_lengths = motion_lengths[:]
    if args.motion_downsample_factor > 1:
        print('Down-sampling motion inputs and targets ...')
        motion_inputs, motion_input_lengths = downsample_sequences(motion_inputs, motion_input_lengths, factor=args.motion_downsample_factor)
        motion_targets, motion_target_lengths = downsample_sequences(motion_targets, motion_target_lengths, factor=args.motion_downsample_factor)
        assert np.all(motion_input_lengths == motion_target_lengths)
        print('done, motion inputs = {}, motion targets = {}'.format(motion_inputs.shape, motion_targets.shape))
        print('')
    assert motion_inputs.shape == motion_targets.shape
    nb_motions, nb_timesteps = motion_inputs.shape[0], motion_inputs.shape[1]
    assert len(motion_input_lengths) == nb_motions
    assert len(motion_target_lengths) == nb_motions

    # Add start and stop signals to motion inputs. Motion inputs are pre-padded, which means that the are
    # right-aligned.
    print('Adding signal to motion inputs ...')
    input_signals = np.zeros((motion_inputs.shape[0], motion_inputs.shape[1], 1))
    for motion, signal, length in zip(motion_inputs, input_signals, motion_input_lengths):
        start_idx = nb_timesteps - length
        stop_idx = nb_timesteps
        signal[start_idx:stop_idx, 0] = 1.
    assert motion_inputs.ndim == input_signals.ndim
    assert motion_inputs.shape[:2] == input_signals.shape[:2]
    old_shape = motion_inputs.shape
    motion_inputs = np.concatenate([motion_inputs, input_signals], axis=2)
    assert motion_inputs.shape == (old_shape[0], old_shape[1], old_shape[2] + 1)
    print('done, motion inputs = {}'.format(motion_inputs.shape))
    print('')

    # Add only stop signal to motion targets. It is the responsibility of the user of the dataset
    # to define a proper start vector for initialization. Motions targets are post-padded, which means
    # that the are left-aligned (e.g. simply using a zero vector).
    print('Adding stop signals to motion targets ...')
    target_signals = np.zeros((motion_targets.shape[0], motion_targets.shape[1], 1))
    for motion, signal, length in zip(motion_targets, target_signals, motion_target_lengths):
        start_idx = 0
        end_idx = length
        signal[start_idx:end_idx, 0] = 1.
    assert motion_targets.ndim == target_signals.ndim
    assert motion_targets.shape[:2] == target_signals.shape[:2]
    old_shape = motion_targets.shape
    motion_targets = np.concatenate([motion_targets, target_signals], axis=2)
    assert motion_targets.shape == (old_shape[0], old_shape[1], old_shape[2] + 1)
    print('done, motion_targets = {}'.format(motion_targets.shape))
    print('')

    assert motion_inputs.shape[:2] == motion_targets.shape[:2]
    return motion_inputs.astype('float32'), motion_targets.astype('float32')


def process_annotations(tokenized_annotations, maxlen, vocabulary, args):
    start_idx = vocabulary.index(args.vocabulary_start_symbol)
    end_idx = vocabulary.index(args.vocabulary_end_symbol)
    unknown_idx = vocabulary.index(args.vocabulary_unknown_symbol)
    padding_idx = vocabulary.index(args.vocabulary_padding_symbol)
    assert padding_idx == 0

    # Vectorize annotations.
    print('Vectorizing annotations ...')
    flattened_tokens = sum(tokenized_annotations, [])
    vectorized_annotations = []
    for tokens in flattened_tokens:
        indexes = []
        for token in tokens:
            if token not in vocabulary:
                idx = unknown_idx
            else:
                idx = vocabulary.index(token)
            indexes.append(idx)
        vectorized_annotations.append(indexes)
    print('done')
    print('')

    # Add start and stop symbols to inputs.
    print('Adding start, stop and padding symbols to annotation inputs ...')
    for annotation in vectorized_annotations:
        annotation.insert(0, start_idx)
        annotation.append(end_idx)
    annotation_inputs = pad_sequences(vectorized_annotations, padding='pre', value=padding_idx, maxlen=maxlen + 2)
    print('done, annotation inputs = {}'.format(annotation_inputs.shape))
    print('')

    # Add stop symbols to targets.
    print('Adding start, stop and padding symbols to annotation targets ...')
    annotation_targets = vectorized_annotations
    for annotation in annotation_targets:
        annotation.append(end_idx)
    annotation_targets = pad_sequences(annotation_targets, padding='post', value=padding_idx, maxlen=maxlen + 1)
    print('done, annotation targets = {}'.format(annotation_targets.shape))
    print('')

    assert annotation_inputs.shape[0] == annotation_targets.shape[0]
    return annotation_inputs.astype('int32'), annotation_targets.astype('int32')
    

def process(motions, tokenized_annotations, ids, maxlen_motions, maxlen_annotations, joint_names, vocabulary, output_path, scaler, args):
    assert len(motions) == len(tokenized_annotations)
    nb_items = len(motions)

    motion_inputs, motion_targets = process_motions(motions, maxlen_motions, args)
    annotation_inputs, annotation_targets = process_annotations(tokenized_annotations, maxlen_annotations, vocabulary, args)

    # Finally, compute the mapping between motion, annotation, and id.
    print('Creating mapping between motion, annotation, and id ...')
    curr_annotation_idx = 0
    mapping = []
    assert len(tokenized_annotations) == len(motions)
    assert len(tokenized_annotations) == len(ids)
    for motion_idx, inner_annotations in enumerate(tokenized_annotations):
        nb_annotations = len(inner_annotations)
        assert nb_annotations >= 1

        # First, iterate over all annotations since a single motion can have multiple annotations.
        for annotation_idx in xrange(curr_annotation_idx, curr_annotation_idx + nb_annotations):
            # Next, iterate over all possible offsets since we have potentially split a single
            # motion into many downsampled versions.
            for motion_offset in xrange(args.motion_downsample_factor):
                actual_motion_idx = nb_items * motion_offset + motion_idx
                assert 0 <= actual_motion_idx < motion_inputs.shape[0]
                assert 0 <= annotation_idx < annotation_inputs.shape[0]
                mapping.append([actual_motion_idx, annotation_idx, motion_idx])
        curr_annotation_idx += nb_annotations
    mapping = np.array(mapping)
    print('done, mapping shape = {}'.format(mapping.shape))
    print('')

    print('Exporting dataset to file "{}" ...'.format(output_path))
    f = h5py.File(output_path, 'w')
    
    f.create_dataset('motion_inputs', data=motion_inputs)
    f['motion_inputs'].attrs['joint_names'] = joint_names
    if scaler is not None:
        f['motion_inputs'].attrs['scaler'] = pickle.dumps(scaler)
    
    f.create_dataset('motion_targets', data=motion_targets)
    f['motion_targets'].attrs['joint_names'] = joint_names
    if scaler is not None:
        f['motion_targets'].attrs['scaler'] = pickle.dumps(scaler)
    
    f.create_dataset('annotation_inputs', data=annotation_inputs)
    f.create_dataset('annotation_targets', data=annotation_targets)
    
    f.create_dataset('vocabulary', data=[v.encode('utf8') for v in vocabulary])
    f['vocabulary'].attrs['padding_symbol'] = args.vocabulary_padding_symbol
    f['vocabulary'].attrs['unknown_symbol'] = args.vocabulary_unknown_symbol
    f['vocabulary'].attrs['start_symbol'] = args.vocabulary_start_symbol
    f['vocabulary'].attrs['end_symbol'] = args.vocabulary_end_symbol
    
    f.create_dataset('ids', data=np.array(ids))
    f.create_dataset('mapping', data=mapping)
    f.close()
    print('done')


def main(args):
    input_path = args.input
    output_path_base = os.path.splitext(input_path)[0]
    
    print('Loading data from file "{}" ...'.format(input_path))
    with open(input_path, 'rb') as f:
        raw_data = pickle.load(f)
    assert len(raw_data['motions']) == len(raw_data['annotations'])
    assert len(raw_data['ids']) == len(raw_data['annotations'])
    motions = []
    annotations = []
    ids = []
    for motion, inner_annotations, id in zip(raw_data['motions'], raw_data['annotations'], raw_data['ids']):
        if len(inner_annotations) == 0:
            continue
        if args.maxlen_motions and len(motion) > args.maxlen_motions:
            continue
        if args.maxlen_annotations and np.max([len(a) for a in inner_annotations]) > args.maxlen_annotations:
            continue
        motions.append(motion)
        annotations.append(inner_annotations)
        ids.append(id)
    joint_names = np.array(raw_data['joint_names'])
    ids = np.array(ids)
    assert len(motions) == len(annotations)
    assert len(motions) == len(ids)
    print('done, {} motions loaded'.format(len(motions)))
    print('')

    print('Scaling motions ...')
    scaler = StandardScaler()
    stacked_motions = np.vstack(motions)
    assert stacked_motions.ndim == 2
    assert stacked_motions.shape[1] == motions[0].shape[1]
    scaler.fit(stacked_motions)
    motions = [scaler.transform(m) for m in motions]
    print('done, min = {}, max = {}'.format(np.min([np.min(m) for m in motions]), np.max([np.max(m) for m in motions])))
    print('')
    
    # Pre-process each annotation.
    replace = {
        u'°': ' degrees',
        'cha cha': 'cha-cha-cha',
        'cha cha cha': 'cha-cha-cha',
        '/ which': ', which',
        'fastly moves': 'moves fast',
        'the person kneeled down is standing up': 'a kneeling person is standing up',
        'tu rns': 'turns',
    }  # TODO: this is pretty hacky
    print('Replacing occurrences of {} in annotations ...'.format(args.blacklisted_tokens))
    for idx, inner_annotations in enumerate(annotations):
        for k, v in replace.iteritems():
            inner_annotations = [annotation.replace(k, v) for annotation in inner_annotations]
        for blacklisted_token in args.blacklisted_tokens:
            inner_annotations = [annotation.replace(blacklisted_token, '') for annotation in inner_annotations]
        annotations[idx] = inner_annotations
    print('done')
    print('')

    # Tokenize annotations. We do this here since we need the same vocabulary over the train and test
    # set.
    print('Tokenizing annotations ...')
    tokenized_annotations = []
    if os.path.exists('corrections.pkl'):
        with open('corrections.pkl', 'rb') as f:
            confirmed_corrections = pickle.load(f)
    else:
        confirmed_corrections = {}
    spell_dict = enchant.Dict(args.dict_name)
    for inner_annotations in annotations:
        inner_tokenized_annotations = []
        for annotation in inner_annotations:
            # TODO: should we really lower-case everything?
            tokens = tokenize(annotation.lower())
            corrected_tokens = []
            for token in tokens:
                if args.disable_spell_check or spell_dict.check(token):
                    corrected_tokens.append(token)
                    continue

                # Ask user to fix.
                if token in confirmed_corrections:
                    correction = confirmed_corrections[token]
                else:
                    suggestions = spell_dict.suggest(token)
                    correction = raw_input('"{}" in sentence "{}" seems to be misspelled. Here are some suggestions: {}\n'.format(token, annotation, suggestions))
                    print('')
                    if not correction:
                        corrected_tokens.append(token)
                        continue
                    correction = correction.lower()
                    confirmed_corrections[token] = correction
                corrected_tokens.extend(tokenize(correction))
            inner_tokenized_annotations.append(corrected_tokens)
        tokenized_annotations.append(inner_tokenized_annotations)
    with open('corrections.pkl', 'wb') as f:
        pickle.dump(confirmed_corrections, f)
    print('done')
    print('')

    # Flatten all tokens and compute vocabulary.
    print('Computing vocabulary ...')
    flattened_tokens = sum(tokenized_annotations, [])
    vocabulary = []
    print('  adding "{}" symbol for padding'.format(args.vocabulary_padding_symbol))
    vocabulary.append(args.vocabulary_padding_symbol)
    print('  adding "{}" symbol for unknown'.format(args.vocabulary_unknown_symbol))
    vocabulary.append(args.vocabulary_unknown_symbol)
    print('  adding "{}" symbol for start of sentence'.format(args.vocabulary_start_symbol))
    vocabulary.append(args.vocabulary_start_symbol)
    print('  adding "{}" symbol for end of sentence'.format(args.vocabulary_end_symbol))
    vocabulary.append(args.vocabulary_end_symbol)
    useful_vocabulary = list(set(sum(flattened_tokens, [])))
    print('  adding {} symbols from tokenized annotations'.format(len(useful_vocabulary)))
    vocabulary += list(set(useful_vocabulary))
    print('  ensuring that padding has index 0 (important for masking!): {}'.format(vocabulary.index(args.vocabulary_padding_symbol)))
    assert vocabulary.index(args.vocabulary_padding_symbol) == 0
    assert len(vocabulary) == len(useful_vocabulary) + 4
    print('done, vocabulary length = {}'.format(len(vocabulary)))
    print('')

    print('Printing vocabulary ...')
    print('  {}'.format('\n  '.join(sorted(vocabulary))))
    if raw_input('Is this okay? (Y/n)') != 'Y':
        exit('stopping')
    print('ok')
    print('')

    # Compute maxlens over data.
    print('Computing lengths ...')
    maxlen_motions = max([len(m) for m in motions])
    if maxlen_motions % args.motion_downsample_factor != 0:
        nb_missing_values = args.motion_downsample_factor - (maxlen_motions % args.motion_downsample_factor)
        maxlen_motions += nb_missing_values
    maxlen_annotations = max([len(a) for a in flattened_tokens])
    print('done, maxlen_motions = {}, maxlen_annotations = {}'.format(maxlen_motions, maxlen_annotations))
    print('')

    # Split & process dataset.
    motions_train_valid, motions_test, tokenized_annotations_train_valid, tokenized_annotations_test, ids_train_valid, ids_test = train_test_split(motions, tokenized_annotations, ids, test_size=args.test_split)
    motions_train, motions_valid, tokenized_annotations_train, tokenized_annotations_valid, ids_train, ids_valid = train_test_split(motions_train_valid, tokenized_annotations_train_valid, ids_train_valid, test_size=args.validation_split)
    process(motions=motions_train, tokenized_annotations=tokenized_annotations_train, ids=ids_train,
        maxlen_motions=maxlen_motions, maxlen_annotations=maxlen_annotations, joint_names=joint_names,
        vocabulary=vocabulary, output_path=output_path_base + '_train.h5f', scaler=scaler, args=args)
    process(motions=motions_valid, tokenized_annotations=tokenized_annotations_valid, ids=ids_valid,
        maxlen_motions=maxlen_motions, maxlen_annotations=maxlen_annotations, joint_names=joint_names,
        vocabulary=vocabulary, output_path=output_path_base + '_valid.h5f', scaler=scaler, args=args)
    process(motions=motions_test, tokenized_annotations=tokenized_annotations_test, ids=ids_test,
        maxlen_motions=maxlen_motions, maxlen_annotations=maxlen_annotations, joint_names=joint_names,
        vocabulary=vocabulary, output_path=output_path_base + '_test.h5f', scaler=scaler, args=args)

    # Save arguments as well.
    with open(output_path_base + '_args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--motion-padding-value', type=float, default=0.)
    parser.add_argument('--motion-downsample-factor', type=int, default=10)
    parser.add_argument('--vocabulary-unknown-symbol', type=str, default='UNK')
    parser.add_argument('--vocabulary-padding-symbol', type=str, default='PAD')
    parser.add_argument('--vocabulary-end-symbol', type=str, default='EOS')
    parser.add_argument('--vocabulary-start-symbol', type=str, default='SOS')
    parser.add_argument('--test-split', type=float, default=.2)
    parser.add_argument('--validation-split', type=float, default=.1)
    parser.add_argument('--dict-name', type=str, default='en_US')
    parser.add_argument('--blacklisted-tokens', type=str, nargs='*', default=['.', ',', '(', ')', ':', ';', '!', '?', '"'])
    parser.add_argument('--disable-spell-check', action='store_true')
    parser.add_argument('--maxlen-motions', type=int, default=None)
    parser.add_argument('--maxlen-annotations', type=int, default=None)
    main(parser.parse_args())
