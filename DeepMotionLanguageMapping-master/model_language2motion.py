import argparse
import os
import cPickle as pickle
import json
import timeit
import subprocess
from tempfile import mkstemp
import warnings

import h5py
import numpy as np; np.random.seed(42)
print(np.__version__)
from tabulate import tabulate
import matplotlib.pyplot as plt

from keras.layers import (Input, LSTM, GRU, Dense, merge, TimeDistributed, RepeatVector,
    Embedding, Lambda, Dropout, BatchNormalization, Bidirectional, Masking)
from keras.models import Model
from keras.utils.generic_utils import Progbar
from keras.callbacks import History, ModelCheckpoint
#from keras.utils.visualize_util import plot
from keras.optimizers import get as get_optimizer
from keras.regularizers import l2
from keras import metrics, objectives
import keras.backend as K
if K.backend() != 'theano':
    raise RuntimeError('only supports Theano backend')

from data_process import tokenize
from visualize_motion_predictions import get_mmm_xml_representation, find_package


TINY = 1e-8


def load_data_predict(path):
    f = h5py.File(path, 'r')
    annotations = f['annotation_inputs']
    motions = f['motion_targets']
    mapping = f['mapping']
    vocabulary = f['vocabulary']
    nb_joints = len(f['motion_targets'].attrs['joint_names'])

    # Collect all motion indexes for a given annotation index.
    motion_indexes_for_annotation_idx = {}
    for motion_idx, annotation_idx, _ in mapping:
        if annotation_idx not in motion_indexes_for_annotation_idx:
            motion_indexes_for_annotation_idx[annotation_idx] = []
        motion_indexes_for_annotation_idx[annotation_idx].append(motion_idx)
    assert len(motion_indexes_for_annotation_idx) == len(annotations)

    # Now, load the actual language data and the corresponding motion references.
    X_language = []
    references = []
    for annotation_idx, motion_indexes in motion_indexes_for_annotation_idx.iteritems():
        X_language.append(annotations[annotation_idx])
        references.append([motions[motion_idx] for motion_idx in motion_indexes])
    assert len(X_language) == len(references)
    X_language = np.array(X_language).astype('int32')

    return X_language, references, nb_joints, vocabulary, f


def load_data_train(path, args):
    f = h5py.File(path, 'r')
    annotations = f['annotation_inputs']
    motions = f['motion_targets']
    mapping = f['mapping']
    nb_vocabulary = f['vocabulary'].shape[0]
    nb_joints = len(f['motion_targets'].attrs['joint_names'])
    
    # Create usable data for training.
    X_language = []
    all_motions = []
    for motion_idx, annotation_idx, _ in mapping:
        X_language.append(annotations[annotation_idx])
        all_motions.append(motions[motion_idx])
    assert len(X_language) == len(all_motions)
    X_language = np.array(X_language)
    Y = np.array(all_motions)

    # Prepare Y representation, that is the representation of the predicted output.
    assert Y.ndim == 3
    if args.motion_representation == 'abs':
        # Use abs representation. This means that the model predicts the absolute value for each joint
        # during each timestep. Since we already have this representation, there's nothing more to do here.
        pass
    elif args.motion_representation == 'diff' or args.motion_representation == 'hybrid':
        # Use diff representation for the targets. This means that the model only predicts the change
        # from the previous configuration, not the absolute joint values.
        old_shape = Y.shape
        pad = np.zeros((Y.shape[0], 1, Y.shape[2]))
        processed_Y = np.hstack([pad, Y])
        assert processed_Y.shape == (old_shape[0], old_shape[1] + 1, old_shape[2])
        processed_Y = np.diff(processed_Y, axis=1)
        assert processed_Y.shape == old_shape
        Y[:, :, :nb_joints] = processed_Y[:, :, :nb_joints]  # only use diff for joints
    
    if args.motion_representation == 'hybrid':
        # In `hybrid` mode, the input is the absolute joint value and the model predicts the change
        # for the next time step.
        X_motion = np.array(all_motions)
    else:
        # In all other cases, the target and the input have the same format.
        X_motion = np.copy(Y)
    # Move motion one back, since this is the previous time step.
    X_motion = X_motion[:, :-1]
    X_motion = np.hstack([np.ones((X_motion.shape[0], 1, X_motion.shape[2])), X_motion])
    
    assert X_motion.shape == Y.shape
    return X_language.astype('int32'), X_motion.astype('float32'), Y.astype('float32'), nb_joints, nb_vocabulary


def load_weights(path, encoder, decoder):
    f = h5py.File(path, 'r')
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    weights = {}
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            weight_values = [g[weight_name] for weight_name in weight_names]
            assert name not in weights
            weights[name] = weight_values
    set_weights(weights, encoder, decoder)


def set_weights(weights, encoder, decoder):
    flattened_layers = encoder.layers[:] + decoder.layers[:]
    names = list(set([l.name for l in flattened_layers]))
    if len(names) != len(flattened_layers):
        raise Exception('The layers of the encoder and decoder contain layers with the same name. Please use unique names.')

    weight_value_tuples = []
    for name, weight_values in weights.iteritems():
        layer = None
        for l in flattened_layers:
            if l.name == name:
                layer = l
                break
        if layer is None:
            raise Exception('The layer "{}", for which we found weights, does not exist'.format(name))
        symbolic_weights = layer.trainable_weights[:] + layer.non_trainable_weights[:]
        if len(weight_values) != len(symbolic_weights):
            raise Exception('Layer #' + str(k) +
                            ' (named "' + layer.name +
                            '" in the current model) was found to '
                            'correspond to layer ' + name +
                            ' in the save file. '
                            'However the new layer ' + layer.name +
                            ' expects ' + str(len(symbolic_weights)) +
                            ' weights, but the saved weights have ' +
                            str(len(weight_values)) +
                            ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)


def dump_data(data, f, args):
    # TODO: implement support for other formats, if necessary
    indent = 4 if not args.disable_pretty_json else None
    json.dump(data, f, indent=indent, sort_keys=True)



def get_rnn_layer(nb_units, return_sequences, stateful, name, dropout_W, dropout_U, args):
    layer = None
    if args.rnn_type == 'lstm':
        layer = LSTM(nb_units, dropout_W=dropout_W, dropout_U=dropout_U, return_sequences=return_sequences,
            stateful=stateful, consume_less=args.consume_less, name=name)
    elif args.rnn_type == 'lstmbn':
        from keras.layers import LSTMBN
        layer = LSTMBN(nb_units, dropout_W=dropout_W, dropout_U=dropout_U, return_sequences=return_sequences,
            stateful=stateful, consume_less=args.consume_less, name=name, batch_norm=True)
    elif args.rnn_type == 'gru':
        layer = GRU(nb_units, dropout_W=dropout_W, dropout_U=dropout_U, return_sequences=return_sequences,
            stateful=stateful, consume_less=args.consume_less, name=name)
    elif args.rnn_type == 'gruln':
        from layers import GRULN
        layer = GRULN(nb_units, dropout_W=dropout_W, dropout_U=dropout_U, return_sequences=return_sequences,
            stateful=stateful, name=name)
    else:
        raise RuntimeError('Unknown RNN type "{}".'.format(args.rnn_type))
    return layer


def build_encoder(x, args):
    dropout_W = args.encoder_dropout[0]
    dropout_U = args.encoder_dropout[1]
    depth = len(args.encoder_units)

    inner_layers = [x]
    for idx, nb_units in enumerate(args.pre_units):
        if args.batch_norm:
            name = 'pre_bn_{}'.format(idx)
            batch_norm = BatchNormalization(name=name)(inner_layers[-1])
            inner_layers.append(batch_norm)

        name = 'pre_{}'.format(idx)
        dense = TimeDistributed(Dense(nb_units, activation=args.pre_activation), name=name)(inner_layers[-1])
        dropout = Dropout(args.pre_dropout)(dense)
        inner_layers.append(dropout)

    for idx, nb_units in enumerate(args.encoder_units):
        # Merge previous output with (transformed) input if peeking is enabled.
        input_layer = None
        if args.encoder_input_peek:
            input_layer = merge([inner_layers[-1], x], mode='concat')
        else:
            input_layer = inner_layers[-1]
        inner_layers.append(input_layer)

        # We currently do not have batch normalization for LSTMs, but at least we can add it in between.
        if args.batch_norm:
           name = 'encoder_bn_{}'.format(idx)
           batch_norm = BatchNormalization(name=name)(inner_layers[-1])
           inner_layers.append(batch_norm)

        # We always return a sequence, except for the last layer.
        if idx == depth - 1:
            return_sequences = False
        else:
            return_sequences = True

        # Define the input from the previous layer(s).
        name = 'encoder_{}'.format(idx)
        encoder_layer = get_rnn_layer(nb_units, return_sequences=return_sequences, stateful=False,
            name=name, dropout_W=dropout_W, dropout_U=dropout_U, args=args)
        if args.bidirectional_encoder:
            encoder_layer = Bidirectional(encoder_layer, merge_mode='concat')
        encoder = encoder_layer(inner_layers[-1])
        inner_layers.append(encoder)
    return inner_layers[-1]


def build_decoder(context, previous_timestep, nb_joints, args, stateful=False):
    # Create the stacked LSTM decoder.
    dropout_W = args.decoder_dropout[0]
    dropout_U = args.decoder_dropout[1]
    depth = len(args.decoder_units)

    inner_layers = []
    if args.inner_units:
        inner_layers.append(merge([context, previous_timestep], mode='concat'))
        for idx, nb_units in enumerate(args.inner_units):
            if args.batch_norm:
               name = 'inner_bn_{}'.format(idx)
               batch_norm = BatchNormalization(name=name)(inner_layers[-1])
               inner_layers.append(batch_norm)

            name = 'inner_{}'.format(idx)
            dense = TimeDistributed(Dense(nb_units, activation=args.inner_activation), name=name)(inner_layers[-1])
            dropout = Dropout(args.inner_dropout)(dense)
            inner_layers.append(dropout)
    else:
        inner_layers.append(previous_timestep)

    rnn_layers = []
    for idx, nb_units in enumerate(args.decoder_units):
        # Define the input from the previous layer(s).
        inputs = [inner_layers[-1]]
        if args.decoder_context_peek:
            inputs.append(context)
        if args.decoder_input_peek:
            inputs.append(previous_timestep)
        input_layer = None
        if len(inputs) > 1:
            input_layer = merge(inputs, mode='concat')
        else:
            input_layer = inputs[0]
        assert input_layer is not None

        # We currently do not have batch normalization for LSTMs, but at least we can add it in between.
        if args.batch_norm:
           name = 'decoder_bn_{}'.format(idx)
           batch_norm = BatchNormalization(name=name)(input_layer)
           input_layer = batch_norm

        name = 'decoder_{}'.format(idx)
        decoder = get_rnn_layer(nb_units, return_sequences=True, stateful=stateful, name=name,
            dropout_W=dropout_W, dropout_U=dropout_U, args=args)(input_layer)
        inner_layers.append(decoder)
        rnn_layers.append(decoder)

    if args.decoder_rnn_peek:
        inner_layers.append(merge(rnn_layers, mode='concat'))

    # Add post layers.
    for idx, nb_units in enumerate(args.post_units):
        if args.batch_norm:
           name = 'post_bn_{}'.format(idx)
           batch_norm = BatchNormalization(name=name)(inner_layers[-1])
           inner_layers.append(batch_norm)

        name = 'post_{}'.format(idx)
        dense = TimeDistributed(Dense(nb_units, activation=args.post_activation), name=name)(inner_layers[-1])
        dropout = Dropout(args.post_dropout)(dense)
        inner_layers.append(dropout)

    if args.batch_norm:
        name = 'output_bn'
        batch_norm = BatchNormalization(name=name)(inner_layers[-1])
        inner_layers.append(batch_norm)

    if args.classifier_dropout > 0.:
        dropout = Dropout(args.classifier_dropout)(inner_layers[-1])
        inner_layers.append(dropout)

    output = None
    if args.decoder == 'normal':
        # We produce the mean and variance for a `nb_joints`-dimensional, diagonal Normal distribution.
        means = TimeDistributed(Dense(nb_joints, activation='linear',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_normal_means')(inner_layers[-1])
        variances = TimeDistributed(Dense(nb_joints, activation='softplus',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_normal_vars')(inner_layers[-1])

        # We also predict an additional channel which indicates when to step.
        stop = TimeDistributed(Dense(1, activation='sigmoid',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_stop')(inner_layers[-1])
        output = merge([means, variances, stop], mode='concat')
    elif args.decoder == 'regression':
        motion_output = TimeDistributed(Dense(nb_joints, activation='linear',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_joints')(inner_layers[-1])
        stop_output = TimeDistributed(Dense(1, activation='sigmoid',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_stop')(inner_layers[-1])
        output = merge([motion_output, stop_output], mode='concat')
    elif args.decoder == 'normal-mixture':
        # We produce the mean and variance for a `nb_joints`-dimensional, diagonal Normal distribution.
        # However, we have multiple distributions that we combine.
        means = TimeDistributed(Dense(nb_joints * args.nb_mixtures, activation='linear',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_mixture_means')(inner_layers[-1])
        variances = TimeDistributed(Dense(nb_joints * args.nb_mixtures, activation='softplus',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_mixture_vars')(inner_layers[-1])
        weights = TimeDistributed(Dense(args.nb_mixtures, activation='softmax',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_mixture_weights')(inner_layers[-1])

        # We also predict an additional channel which indicates when to step.
        stop = TimeDistributed(Dense(1, activation='sigmoid',
            W_regularizer=l2(args.classifier_l2_regularizer)), name='dense_stop')(inner_layers[-1])
        output = merge([means, variances, weights, stop], mode='concat')

    assert output is not None
    return output


def build_stateful_encoder_decoder_model(input_length, nb_vocabulary, nb_joints, batch_size, args):
    # The input are language sequences of shape (nb_samples, nb_steps).
    language_input = Input(shape=(input_length,), dtype='int32', name='language_input')
    embedded_language = Embedding(nb_vocabulary, args.embedding_size, input_length=input_length,
        name='embedding_language', dropout=args.embedding_dropout, mask_zero=True)(language_input)
    encoder = build_encoder(embedded_language, args)
    model_encoder = Model(input=language_input, output=encoder)

    # Next, decode the sequences.
    context_input = Input(batch_shape=(batch_size, 1, encoder._keras_shape[-1]), name='context_input')
    previous_input = Input(batch_shape=(batch_size, 1, nb_joints + 1), name='previous_input')
    if args.motion_input_masking:
        masked_previous_input = Masking()(previous_input)
    else:
        masked_previous_input = previous_input
    decoder = build_decoder(context_input, previous_input, nb_joints, args, stateful=True)
    model_decoder = Model(input=[context_input, previous_input], output=decoder)

    return model_encoder, model_decoder


def serializable_args(args):
    args_data = vars(args)
    del args_data['func']
    return args_data


def prepare_output(args):
    # Ensure that `output` points to a directory. If possible, create it. Otherwise, exit.
    output_path = args.output
    if not os.path.exists(output_path):
        print('Creating output directory "{}" ...'.format(output_path))
        os.makedirs(output_path)
        print('done')
        print('')
    elif os.path.isdir(output_path):
        print('Using existing output directory "{}"'.format(output_path))
        print('')
    else:
        exit('Output directory points to "{}", but a file with that name already exists'.format(output_path))
    return output_path


def normal_loss(y_true, y_pred):
    assert normal_loss.nb_joints
    nb_joints = normal_loss.nb_joints

    means = y_pred[:, :, :nb_joints]
    variances = y_pred[:, :, nb_joints:2 * nb_joints] + TINY
    stop = y_pred[:, :, -1]
    y_true_motion = y_true[:, :, :nb_joints]
    y_true_stop = y_true[:, :, -1]

    gaussian_pdf = 1. / K.sqrt(variances * 2. * np.pi) * K.exp(-K.square(y_true_motion - means) / (2. * variances))
    # pdf of multivariate diagonal Gaussian is product of individual Gaussian components. We use the
    # log pdf, hence sum over the last axis (= all joints)
    log_gaussian_pdf = K.sum(K.log(gaussian_pdf + TINY), axis=-1)

    bernoulli_pdf = y_true_stop * stop + (1. - y_true_stop) * (1. - stop)
    log_bernoulli_pdf = K.log(bernoulli_pdf + TINY)

    loss = -(log_gaussian_pdf + log_bernoulli_pdf)
    return loss


def normal_mixture_surrogate_loss(y_true, y_pred):
    assert normal_mixture_surrogate_loss.args
    assert normal_mixture_surrogate_loss.nb_joints
    args = normal_mixture_surrogate_loss.args
    nb_mixtures = args.nb_mixtures
    nb_joints = normal_mixture_surrogate_loss.nb_joints
    
    all_means = y_pred[:, :, :nb_joints * nb_mixtures]
    all_variances = y_pred[:, :, nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures] + TINY
    weights = y_pred[:, :, 2 * nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures + nb_mixtures]
    stop = y_pred[:, :, -1]
    y_true_motion = y_true[:, :, :nb_joints]
    y_true_stop = y_true[:, :, -1]

    log_mixture_pdf = None
    for mixture_idx in xrange(nb_mixtures):
        start_idx = mixture_idx * nb_joints
        means = all_means[:, :, start_idx:start_idx + nb_joints]
        variances = all_variances[:, :, start_idx:start_idx + nb_joints]
        pdf = 1. / K.sqrt(variances * 2. * np.pi) * K.exp(-K.square(y_true_motion - means) / (2. * variances))
        weighted_pdf = weights[:, :, mixture_idx] * K.sum(K.log(pdf + TINY), axis=-1)
        if log_mixture_pdf is None:
            log_mixture_pdf = weighted_pdf
        else:
            log_mixture_pdf += weighted_pdf
    assert log_mixture_pdf is not None
    
    bernoulli_pdf = y_true_stop * stop + (1. - y_true_stop) * (1. - stop)
    log_bernoulli_pdf = K.log(bernoulli_pdf + TINY)

    if args.mixture_regularizer_type == 'cv':
        # We want to use (std / mean)^2 = std^2 / mean^2 = var / mean^2.
        mixture_reg = K.var(weights, axis=-1) / K.square(K.mean(weights, axis=-1))
    elif args.mixture_regularizer_type == 'l2':
        mixture_reg = K.sum(K.square(weights), axis=-1)
    else:
        mixture_reg = 0.
    loss = -(log_mixture_pdf + log_bernoulli_pdf) + args.mixture_regularizer * mixture_reg
    return loss


def normal_mixture_loss(y_true, y_pred):
    assert normal_mixture_loss.args.nb_mixtures
    assert normal_mixture_loss.nb_joints
    nb_mixtures = normal_mixture_loss.args.nb_mixtures
    nb_joints = normal_mixture_loss.nb_joints
    
    all_means = y_pred[:, :, :nb_joints * nb_mixtures]
    all_variances = y_pred[:, :, nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures] + TINY
    weights = y_pred[:, :, 2 * nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures + nb_mixtures]
    stop = y_pred[:, :, -1]
    y_true_motion = y_true[:, :, :nb_joints]
    y_true_stop = y_true[:, :, -1]

    mixture_pdf = None
    for mixture_idx in xrange(nb_mixtures):
        start_idx = mixture_idx * nb_joints
        means = all_means[:, :, start_idx:start_idx + nb_joints]
        variances = all_variances[:, :, start_idx:start_idx + nb_joints]
        pdf = 1. / K.sqrt(variances * 2. * np.pi) * K.exp(-K.square(y_true_motion - means) / (2. * variances))
        weighted_pdf = weights[:, :, mixture_idx] * K.prod(pdf + TINY, axis=-1)
        if mixture_pdf is None:
            mixture_pdf = weighted_pdf
        else:
            mixture_pdf += weighted_pdf
    log_mixture_pdf = K.log(mixture_pdf + TINY)
    assert log_mixture_pdf is not None
    
    bernoulli_pdf = y_true_stop * stop + (1. - y_true_stop) * (1. - stop)
    log_bernoulli_pdf = K.log(bernoulli_pdf + TINY)

    loss = -(log_mixture_pdf + log_bernoulli_pdf)
    return loss


def normal_mixture_mean_variances(y_true, y_pred):
    assert normal_mixture_mean_variances.args.nb_mixtures
    assert normal_mixture_mean_variances.nb_joints
    nb_mixtures = normal_mixture_mean_variances.args.nb_mixtures
    nb_joints = normal_mixture_mean_variances.nb_joints

    y_pred_variances = y_pred[:, :, nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures]
    y_pred_weights = y_pred[:, :, 2 * nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures + nb_mixtures]

    variances = None
    for mixture_idx in xrange(nb_mixtures):
        start_idx = mixture_idx * nb_joints
        w = K.repeat_elements(y_pred_weights[:, :, mixture_idx:mixture_idx+1], nb_joints, axis=-1)
        v = y_pred_variances[:, :, start_idx:start_idx + nb_joints]
        weighted_variances = w * v
        if variances is None:
            variances = weighted_variances
        else:
            variances += weighted_variances
    return K.mean(variances)


def motion_mae(y_true, y_pred):
    assert motion_mae.nb_joints
    nb_joints = motion_mae.nb_joints

    y_pred_means = y_pred[:, :, :nb_joints]
    y_true_motion = y_true[:, :, :nb_joints]
    return metrics.mae(y_true_motion, y_pred_means)


def normal_mean_variances(y_true, y_pred):
    assert normal_mean_variances.nb_joints
    nb_joints = normal_mean_variances.nb_joints

    variances = y_pred[:, :, nb_joints:2 * nb_joints]
    return K.mean(variances)


def motion_mixture_mae(y_true, y_pred):
    assert motion_mixture_mae.args.nb_mixtures
    assert motion_mixture_mae.nb_joints
    nb_mixtures = motion_mixture_mae.args.nb_mixtures
    nb_joints = motion_mixture_mae.nb_joints

    y_pred_means = y_pred[:, :, :nb_joints * nb_mixtures]
    y_pred_weights = y_pred[:, :, 2 * nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures + nb_mixtures]
    y_true_motion = y_true[:, :, :nb_joints]
    
    # This is an estimate of how well the model fits the true motion. We estimate this by
    # computing the mean absolute error for each mixture and weighting it accordingly.
    mae = None
    for mixture_idx in xrange(nb_mixtures):
        start_idx = mixture_idx * nb_joints
        w = K.repeat_elements(y_pred_weights[:, :, mixture_idx:mixture_idx+1], nb_joints, axis=-1)
        m = y_pred_means[:, :, start_idx:start_idx + nb_joints]
        weighted_mae = w * K.abs(y_true_motion - m)
        if mae is None:
            mae = weighted_mae
        else:
            mae += weighted_mae
    return K.mean(mae)


def mean_mixture(y_true, y_pred):
    assert mean_mixture.args.nb_mixtures
    assert mean_mixture.nb_joints
    nb_mixtures = mean_mixture.args.nb_mixtures
    nb_joints = mean_mixture.nb_joints

    y_pred_weights = y_pred[:, :, 2 * nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures + nb_mixtures]
    return K.mean(K.max(y_pred_weights, axis=-1))


def std_mixture(y_true, y_pred):
    assert std_mixture.args.nb_mixtures
    assert std_mixture.nb_joints
    nb_mixtures = std_mixture.args.nb_mixtures
    nb_joints = std_mixture.nb_joints

    y_pred_weights = y_pred[:, :, 2 * nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures + nb_mixtures]
    return K.std(K.max(y_pred_weights, axis=-1))


def mean_mixture_idx(y_true, y_pred):
    assert mean_mixture.args.nb_mixtures
    assert mean_mixture.nb_joints
    nb_mixtures = mean_mixture.args.nb_mixtures
    nb_joints = mean_mixture.nb_joints

    y_pred_weights = y_pred[:, :, 2 * nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures + nb_mixtures]
    return K.mean(K.argmax(y_pred_weights, axis=-1))


def std_mixture_idx(y_true, y_pred):
    assert std_mixture_idx.args.nb_mixtures
    assert std_mixture_idx.nb_joints
    nb_mixtures = std_mixture_idx.args.nb_mixtures
    nb_joints = std_mixture_idx.nb_joints

    y_pred_weights = y_pred[:, :, 2 * nb_joints * nb_mixtures:2 * nb_joints * nb_mixtures + nb_mixtures]
    return K.std(K.argmax(y_pred_weights, axis=-1))


def stop_mae(y_true, y_pred):
    y_pred_stop = y_pred[:, :, -1]
    y_true_stop = y_true[:, :, -1]
    return metrics.mae(y_true_stop, y_pred_stop)


def data_generator(X_language, X_motion, Y, nb_vocabulary, nb_joints, args):
    nb_epoch = 0
    nb_samples = X_language.shape[0]
    nb_batches = int(np.ceil(nb_samples / float(args.batch_size)))

    if args.mask_target:
        # Find the stop signal for each output and create the weighting matrix.
        candidates = np.where(Y[:, :, -1] == 1.)
        stop_indexes = [0 for _ in xrange(nb_samples)]
        for idx0, idx1 in zip(*candidates):
            stop_indexes[idx0] = max(idx1, stop_indexes[idx0])
        assert len(stop_indexes) == nb_samples
        sample_weights = np.zeros((Y.shape[0], Y.shape[1]))  # (nb_samples, nb_steps)
        for idx0, idx1 in enumerate(stop_indexes):
            # We add a small offset to the stop index to ensure that we learn to properly end a motion.
            sample_weights[idx0, :idx1 + args.mask_target_offset] = 1.
    else:
        sample_weights = None

    while True:
        perm = np.random.permutation(nb_samples)
        
        # Process batch by batch.
        for batch_idx in xrange(nb_batches):
            print('idx',batch_idx)
            start_idx = batch_idx * args.batch_size
            indexes = perm[start_idx:start_idx + args.batch_size]
            batch_X_l = X_language[indexes]
            batch_X_m = X_motion[indexes]
            batch_Y = Y[indexes]
            if sample_weights is not None:
                batch_weights = sample_weights[indexes]
                yield ([batch_X_l, batch_X_m], batch_Y, batch_weights)
            else:
                yield ([batch_X_l, batch_X_m], batch_Y)
        nb_epoch += 1


def prepare_for_training(output_path, args, include_metrics=True):
    # Load training data.
    print('Loading training data "{}" ...'.format(args.input))
    X_language_train, X_motion_train, Y_train, nb_joints, nb_vocabulary = load_data_train(args.input, args)
    print('done, X_language_train = {}, X_motion_train= {}, Y_train = {}, nb_joints = {}, nb_vocabulary = {}'.format(X_language_train.shape, X_motion_train.shape, Y_train.shape, nb_joints, nb_vocabulary))
    print('')

    # Load validation data, if applicable.
    if args.validation_input:
        print('Loading validation data "{}" ...'.format(args.validation_input))
        X_language_valid, X_motion_valid, Y_valid, nb_joints_valid, nb_vocabulary_valid = load_data_train(args.validation_input, args)
        assert nb_vocabulary == nb_vocabulary_valid
        assert nb_joints == nb_joints_valid
        print('done, X_language_valid = {}, X_motion_valid = {}, Y_valid = {}, nb_joints = {}, nb_vocabulary = {}'.format(X_language_valid.shape, X_motion_valid.shape, Y_valid.shape, nb_joints_valid, nb_vocabulary_valid))
        print('')

    # The input are annotation sequences of shape (nb_samples, nb_steps). We first learn an embedding.
    language_input = Input(shape=(X_language_train.shape[1],), dtype='int32', name='language_input')
    embedded_language = Embedding(nb_vocabulary, args.embedding_size, input_length=X_language_train.shape[1],
        name='embedding_language', dropout=args.embedding_dropout, mask_zero=True)(language_input)

    # The other input are motion sequences of the previous step of shape (nb_samples, nb_steps, nb_dim). During
    # training, we set this to the ground truth.
    motion_input = Input(shape=X_motion_train.shape[1:], name='motion_input')
    if args.motion_input_masking:
        motion_input_processed = Masking()(motion_input)
    else:
        motion_input_processed = motion_input

    # Create the encoder and decoder.
    encoder = build_encoder(embedded_language, args)
    repeated_context = RepeatVector(Y_train.shape[1])(encoder)
    decoder = build_decoder(repeated_context, motion_input_processed, nb_joints, args)

    # Create the model.
    print('Compiling the model ...')
    model = Model(input=[language_input, motion_input], output=decoder)
    print(model.summary())
    optimizer = get_optimizer(args.optimizer)
    K.set_value(optimizer.lr, args.lr)
    if args.clipnorm is not None:
        optimizer.clipnorm = args.clipnorm
    if args.clipvalue is not None:
        optimizer.clipvalue = args.clipvalue
    loss = None
    metrics = None
    if args.decoder == 'normal':
        loss = normal_loss
        metrics = [motion_mae, stop_mae, normal_mean_variances]
    elif args.decoder == 'normal-mixture':
        if args.surrogate_loss:
            loss = normal_mixture_surrogate_loss
        else:
            loss = normal_mixture_loss
        metrics = [motion_mixture_mae, stop_mae, mean_mixture, std_mixture, mean_mixture_idx, std_mixture_idx, normal_mixture_mean_variances]
    elif args.decoder == 'regression':
        loss = 'mse'
        metrics = [motion_mae, stop_mae]
    if getattr(args, 'mask_target', False):
        sample_weight_mode = 'temporal'
    else:
        sample_weight_mode = None
    for metric in metrics:
        metric.nb_joints = nb_joints
        metric.args = args
    loss.nb_joints = nb_joints
    loss.args = args
    if not include_metrics:
        metrics = None
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, sample_weight_mode=sample_weight_mode)
    print('done')
    print('')

    data_train = (X_language_train, X_motion_train, Y_train, nb_joints, nb_vocabulary)
    data_valid = None
    if args.validation_input:
        data_valid = ([X_language_valid, X_motion_valid], Y_valid)
    return data_train, data_valid, model, optimizer


def train(args):
    output_path = prepare_output(args)
    train_data, valid_data, model, optimizer = prepare_for_training(output_path, args)
    X_language_train, X_motion_train, Y_train, nb_joints, nb_vocabulary = train_data
    
    # Save model information to output.
    print('Saving model information to "{}" ...'.format(output_path))
    with open(os.path.join(output_path, 'model.json'), 'w') as f:
        f.write(model.to_json())
    #plot(model, to_file=os.path.join(output_path, 'model.pdf'), show_shapes=True)
    #plot(model, to_file=os.path.join(output_path, 'model.dot'), show_shapes=True)
    with open(os.path.join(output_path, 'train_args.json'), 'w') as f:
        dump_data(serializable_args(args), f, args)
    with open(os.path.join(output_path, 'train_optimizer.json'), 'w') as f:
        dump_data(optimizer.get_config(), f, args)
    print('done')
    print('')

    # Start the training and save weights once done or aborted. We use our own history callback
    # since training might be aborted, in which case `fit` does not return.
    history = History()
    checkpoint = ModelCheckpoint(filepath=os.path.join(output_path, 'model_weights_{epoch:03d}.h5f'),
        save_weights_only=True)
    start = timeit.default_timer()
    print('Training ...')
    try:
        train_gen = data_generator(X_language_train, X_motion_train, Y_train, nb_vocabulary, nb_joints, args)
        print(type(train_gen))
        valid_gen, nb_val_samples = None, None
        if valid_data:
            print('valid data')
            (X_language_valid, X_motion_valid), Y_valid = valid_data
            valid_gen = data_generator(X_language_valid, X_motion_valid, Y_valid, nb_vocabulary, nb_joints, args)
            nb_val_samples = X_language_valid.shape[0]
        samples_per_epoch = X_language_train.shape[0]
        print('start model.fit_generator')
        model.fit_generator(train_gen, samples_per_epoch, nb_epoch=args.nb_epoch, validation_data=valid_gen,
            callbacks=[history, checkpoint], nb_val_samples=nb_val_samples)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print('error',e)
    finally:
        duration = timeit.default_timer() - start
        print('')
        print('done, took {}s'.format(duration))
        print('')

        print('Saving training results to "{}" ...').format(output_path)
        # Save weights first, since this is the most important result.
        model.save_weights(os.path.join(output_path, 'model_weights.h5f'), overwrite=True)

        # Save training history.
        history_data = dict(history.history)
        history_data['nb_epoch'] = len(history.epoch)
        history_data['duration'] = duration
        with open(os.path.join(output_path, 'train_history.json'), 'w') as f:
            dump_data(history_data, f, args)
        print('done')


def evaluate(args):
    output_path = prepare_output(args)
    train_data, valid_data, model, optimizer = prepare_for_training(output_path, args, include_metrics=False)
    X_language_train, X_motion_train, Y_train, nb_joints, nb_vocabulary = train_data

    all_weights = [os.path.join(args.model, p) for p in os.listdir(args.model) if os.path.splitext(p)[1].lower() == '.h5f']
    epochs = []
    valid_weights = []
    for w in all_weights:
        try:
            epoch = int(os.path.splitext(w.split('_')[-1])[0])
        except:
            continue
        epochs.append(epoch)
        valid_weights.append(w)
    sorted_indexes = np.argsort(epochs)
    epochs = [epochs[idx] for idx in sorted_indexes]
    valid_weights = [valid_weights[idx] for idx in sorted_indexes]

    # Evaluate each epoch.
    train_metrics = []
    valid_metrics = []
    try:
        for idx, weights in enumerate(valid_weights):
            print('{}/{}: Evaluating "{}" ...'.format(idx + 1, len(valid_weights), weights))
            model.load_weights(weights)
            metrics = model.evaluate([X_language_train, X_motion_train], Y_train, batch_size=args.batch_size)
            train_metrics.append(metrics if isinstance(metrics, (list, tuple)) else [metrics])
            print(zip(model.metrics_names, train_metrics[-1]))
            if valid_data:
                X_language_valid, X_motion_valid, Y_valid = valid_data
                metrics = model.evaluate([X_language_valid, X_motion_valid], Y_valid, batch_size=args.batch_size)
                valid_metrics.append(metrics if isinstance(metrics, (list, tuple)) else [metrics])
                print(zip(model.metrics_names, valid_metrics[-1]))
            print('done')
            print('')
    except KeyboardInterrupt:
        pass
    train_metrics = np.array(train_metrics)
    valid_metrics = np.array(valid_metrics)

    print('Exporting results to "{}" ...'.format(output_path))
    data = {}
    for idx, name in enumerate(model.metrics_names):
        data[name] = train_metrics[:, idx].tolist()
        if valid_data:
            data['val_' + name] = valid_metrics[:, idx].tolist()
    data['epochs'] = epochs
    with open(os.path.join(output_path, 'evaluation.json'), 'w') as f:
        dump_data(data, f, args)
    print('done')


def perform_regression(preds, decoder, nb_joints, previous_outputs, log_probabilities, done, args):
    preds = preds.reshape(preds.shape[0], nb_joints + 1)
    samples = preds[:, :nb_joints]
    stops = preds[:, -1]
    for idx, (sample, stop) in enumerate(zip(samples, stops)):
        if done[idx]:
            continue
        combined = np.concatenate([sample, [stop]])
        assert combined.shape == (nb_joints + 1,)
        previous_outputs[idx].append(combined)
        log_probabilities[idx] += 0.  # TODO: compute me!
        if stop < .5:
           done[idx] = True

    return previous_outputs, log_probabilities, done


def gaussian_pdf(sample, means, variances):
    assert sample.ndim == 1
    assert sample.shape == means.shape
    assert sample.shape == variances.shape

    return 1. / (np.sqrt(2. * np.pi * variances)) * np.exp(-np.square(sample - means) / (2. * variances))


def bernoulli_pdf(sample, p):
    return float(sample) * p + float(1. - sample) * (1. - p)


def perform_normal_sampling(preds, decoder, nb_joints, previous_outputs, log_probabilities, done, args):
    preds = preds.reshape(preds.shape[0], 2 * nb_joints + 1)
    means = preds[:, :nb_joints]
    variances = preds[:, nb_joints:2 * nb_joints]
    stops = preds[:, -1]
    
    # Sample joint values.
    samples = np.random.normal(means, variances)
    assert samples.shape == (preds.shape[0], nb_joints)
    
    for idx, (sample, stop) in enumerate(zip(samples, stops)):
        if done[idx]:
            continue

        sampled_stop = np.random.binomial(n=1, p=stop)
        combined = np.concatenate([sample, [sampled_stop]])
        assert combined.shape == (nb_joints + 1,)
        previous_outputs[idx].append(combined)
        log_probabilities[idx] += np.sum(np.log(gaussian_pdf(sample, means[idx], variances[idx])))
        log_probabilities[idx] += np.log(bernoulli_pdf(sampled_stop, stop))
        done[idx] = (sampled_stop == 0)

    return previous_outputs, log_probabilities, done


def perform_normal_mixture_sampling(preds, decoder, nb_joints, previous_outputs, log_probabilities, done, args):
    preds = preds.reshape(preds.shape[0], 2 * nb_joints * args.nb_mixtures + args.nb_mixtures + 1)
    all_means = preds[:, :nb_joints * args.nb_mixtures]
    all_variances = preds[:, nb_joints * args.nb_mixtures:2 * nb_joints * args.nb_mixtures] + TINY
    weights = preds[:, 2 * nb_joints * args.nb_mixtures:2 * nb_joints * args.nb_mixtures + args.nb_mixtures]
    assert all_means.shape[-1] == nb_joints * args.nb_mixtures
    assert all_variances.shape[-1] == nb_joints * args.nb_mixtures
    assert weights.shape[-1] == args.nb_mixtures
    stops = preds[:, -1]
    
    # Sample joint values.
    samples = np.zeros((preds.shape[0], nb_joints))
    means = np.zeros((preds.shape[0], nb_joints))
    variances = np.zeros((preds.shape[0], nb_joints))
    for width_idx in xrange(preds.shape[0]):
        # Decide which mixture to sample from
        p = weights[width_idx]
        assert p.shape == (args.nb_mixtures,)
        mixture_idx = np.random.choice(range(args.nb_mixtures), p=p)

        # Sample from it.
        start_idx = mixture_idx * nb_joints
        m = all_means[width_idx, start_idx:start_idx + nb_joints]
        v = all_variances[width_idx, start_idx:start_idx + nb_joints]
        assert m.shape == (nb_joints,)
        assert m.shape == v.shape
        s = np.random.normal(m, v)
        samples[width_idx, :] = s
        means[width_idx, :] = m
        variances[width_idx, :] = v
    
    for idx, (sample, stop) in enumerate(zip(samples, stops)):
        if done[idx]:
            continue

        sampled_stop = np.random.binomial(n=1, p=stop)
        combined = np.concatenate([sample, [sampled_stop]])
        assert combined.shape == (nb_joints + 1,)
        previous_outputs[idx].append(combined)
        log_probabilities[idx] += np.sum(np.log(gaussian_pdf(sample, means[idx], variances[idx])))
        log_probabilities[idx] += np.log(bernoulli_pdf(sampled_stop, stop))
        done[idx] = (sampled_stop == 0)

    return previous_outputs, log_probabilities, done


def decode(context, decoder, nb_joints, language, references, args, init=None):
    # Prepare data structures for graph search.
    if init is None:
        init = np.ones(nb_joints + 1)
    assert init.shape == (nb_joints + 1,)
    previous_outputs = [[np.copy(init)] for _ in xrange(args.width)]
    repeated_context = np.repeat(context.reshape(1, context.shape[-1]), args.width, axis=0)
    repeated_context = repeated_context.reshape(args.width, 1, context.shape[-1])
    log_probabilities = [0. for _ in xrange(args.width)]
    done = [False for _ in xrange(args.width)]

    # Reset the decoder.
    decoder.reset_states()

    # Iterate over time.
    predictions = [[] for _ in range(args.width)]
    for _ in xrange(args.depth):
        previous_output = np.array([o[-1] for o in previous_outputs])
        assert previous_output.ndim == 2
        previous_output = previous_output.reshape((previous_output.shape[0], 1, previous_output.shape[1]))
        preds = decoder.predict_on_batch([repeated_context, previous_output])
        assert preds.shape[0] == args.width
        for idx, (pred, d) in enumerate(zip(preds, done)):
            if d:
                continue
            predictions[idx].append(pred)

        # Perform actual decoding.
        if args.decoder == 'normal':
            fn = perform_normal_sampling
        elif args.decoder == 'regression':
            fn = perform_regression
        elif args.decoder == 'normal-mixture':
            fn = perform_normal_mixture_sampling
        else:
            fn = None
            raise ValueError('Unknown decoder "{}"'.format(args.decoder))
        previous_outputs, log_probabilities, done = fn(preds, decoder, nb_joints, previous_outputs, log_probabilities, done, args)

        if args.motion_representation == 'hybrid':
            # For each element of the beam, add the new delta (index -1) to the previous element (index -2)
            # to obtain the absolute motion.
            for po in previous_outputs:
                po[-1][:nb_joints] = po[-2][:nb_joints] + po[-1][:nb_joints]
        
        # Check if we're done before reaching `args.depth`.
        if np.all(done):
            break

    # Convert to numpy arrays.
    predictions = [np.array(preds)[:, 0, :].astype('float32') for preds in predictions]
    hypotheses = []
    for previous_output in previous_outputs:
        motion = np.array(previous_output)[1:].astype('float32')  # remove init state
        if args.motion_representation == 'diff':
            motion[:, :nb_joints] = np.cumsum(motion[:, :nb_joints], axis=0)
        assert motion.shape[-1] == nb_joints + 1
        hypotheses.append(motion.astype('float32'))
    
    # Record data.
    data = {
        'hypotheses': hypotheses,
        'log_probabilities': log_probabilities,
        'references': references,
        'language': language,
        'predictions': predictions,
    }
    return data


def predict(args):
    output_path = prepare_output(args)

    # Load data.
    print('Loading data "{}" ...'.format(args.input))
    X, references, nb_joints, vocabulary, h5f = load_data_predict(args.input)
    nb_vocabulary = len(vocabulary)
    start_idx = list(vocabulary).index(vocabulary.attrs['start_symbol'])
    print('done, X = {}, references = {}, nb_joints = {}, nb_vocabulary = {}'.format(X.shape, len(references), nb_joints, nb_vocabulary))
    print('')

    # Compile both models. We don't use the optimizer, so we specify anything here.
    print('Compiling models and loading weights "{}" ...'.format(args.model))
    model_encoder, model_decoder = build_stateful_encoder_decoder_model(input_length=X.shape[1],
        nb_vocabulary=nb_vocabulary, nb_joints=nb_joints, batch_size=args.width, args=args)
    model_encoder.compile(optimizer='adam', loss='mse')
    model_decoder.compile(optimizer='adam', loss='mse')
    load_weights(args.model, encoder=model_encoder, decoder=model_decoder)
    print(model_encoder.summary())
    print(model_decoder.summary())
    print('done')
    print('')

    # Start by predicting the context vectors.
    print('Computing context vectors ...')
    contexts = model_encoder.predict(X, batch_size=args.batch_size, verbose=1)
    print('done, contexts shape = {}'.format(contexts.shape))
    print('')

    # Decode motions.
    print('Decoding {} context vectors using "{}" ...'.format(contexts.shape[0], args.decoder))
    decoded_data = []
    progbar = Progbar(target=contexts.shape[0])
    try:
        for sample_idx, context in enumerate(contexts):
            language = []
            for token in list(reversed(X[sample_idx]))[1:]:
                if token == start_idx:
                    break
                language.append(vocabulary[token])
            language = ' '.join(language)
            data = decode(context, decoder=model_decoder, nb_joints=nb_joints, language=language, references=references[sample_idx], args=args)
            if not args.include_raw_predictions:
                # Remove raw predictions to keep file size manageable.
                del data['predictions']
            decoded_data.append(data)

            # Update UI.
            progbar.update(sample_idx)
    except KeyboardInterrupt:
        pass
    print('')
    print('done')
    print('')

    print('Saving results in "{}" ...'.format(output_path))
    with h5py.File(os.path.join(output_path, 'predict_contexts.h5f'), 'w') as f:
        f.create_dataset('contexts', data=contexts)
    with open(os.path.join(output_path, 'predict.pkl'), 'wb') as f:
        data = {
            'decoded_data': decoded_data,
            'joint_names': h5f['motion_inputs'].attrs['joint_names'],
            'scaler': h5f['motion_inputs'].attrs['scaler'],
            'nb_mixtures': args.nb_mixtures,
            'decoder': args.decoder,
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_path, 'predict_args.json'), 'w') as f:
        dump_data(serializable_args(args), f, args)
    print('done')


def interactive(args):
    # Find necessary MMM files.
    try:
        mmm_root_path = find_package('MMMTools')
        if not mmm_root_path:
            exit('could not find MMMTools')
        model_path = os.path.join(mmm_root_path, 'data', 'Model', 'Winter', 'mmm.xml')
        bin_path = os.path.join(mmm_root_path, 'build', 'bin', 'MMMViewer')
    except:
        model_path, bin_path = None, None

    print('Loading data "{}" ...'.format(args.input))
    f = h5py.File(args.input, 'r')
    vocabulary = list(f['vocabulary'])
    pad_symbol = f['vocabulary'].attrs['padding_symbol']
    unknown_symbol = f['vocabulary'].attrs['unknown_symbol']
    start_symbol = f['vocabulary'].attrs['start_symbol']
    end_symbol = f['vocabulary'].attrs['end_symbol']
    nb_vocabulary = len(vocabulary)
    joint_names = list(f['motion_targets'].attrs['joint_names'])
    nb_joints = len(joint_names)
    if 'scaler' in f['motion_inputs'].attrs:
        scaler = pickle.loads(f['motion_inputs'].attrs['scaler'])
    else:
        scaler = None
    f.close()
    print('done, nb_joints = {}, nb_vocabulary = {}'.format(nb_joints, nb_vocabulary))
    print('')

    # Compile both models. We don't use the optimizer, so we specify anything here.
    print('Compiling models and loading weights "{}" ...'.format(args.model))
    model_encoder, model_decoder = build_stateful_encoder_decoder_model(input_length=args.maxlen,
        nb_vocabulary=nb_vocabulary, nb_joints=nb_joints, batch_size=args.width, args=args)
    model_encoder.compile(optimizer='adam', loss='mse')
    model_decoder.compile(optimizer='adam', loss='mse')
    load_weights(args.model, encoder=model_encoder, decoder=model_decoder)
    print(model_encoder.summary())
    print(model_decoder.summary())
    print('done')
    print('')

    while True:
        query = raw_input('Enter a motion description: ').strip()
        if len(query) == 0:
            break
        tokens = tokenize(query.lower())
        token_indexes = [vocabulary.index(t) if t in vocabulary else vocabulary.index(unknown_symbol) for t in tokens]
        processed_input = [vocabulary[idx] for idx in token_indexes]
        final_token_indexes = [vocabulary.index(start_symbol)] + token_indexes + [vocabulary.index(end_symbol)]
        while len(final_token_indexes) < args.maxlen:
            final_token_indexes.insert(0, vocabulary.index(pad_symbol))
        assert len(final_token_indexes) == args.maxlen
        print('  processed query: {}'.format(processed_input))
        print('  query tokens:    {}'.format(','.join([str(x) for x in final_token_indexes])))

        print('  computing context vector ...')
        context = model_encoder.predict(np.array([final_token_indexes]), batch_size=1)
        assert context.shape[0] == 1
        print('  decoding context vector ...')
        init = np.ones(nb_joints + 1)
        data = decode(context, decoder=model_decoder, nb_joints=nb_joints, language=None, references=None, args=args, init=init)
        best_idx = np.argmax(data['log_probabilities'])
        hypothesis = data['hypotheses'][best_idx]
        log_proba = data['log_probabilities'][best_idx]
        print('  log likelihood: {} (mean: {}+-{})'.format(log_proba, np.mean(data['log_probabilities']), np.std(data['log_probabilities'])))

        # Truncate first (initialization) and last (EOF) element.
        hypothesis = hypothesis[1:, :][:-1, :]

        if args.decoder == 'normal-mixture':
            # Extract mixture components and plot heat map over time.
            weights = data['predictions'][best_idx][:, -args.nb_mixtures-1:-1]
            print('  weights mean: {}'.format(np.mean(np.max(weights, axis=-1))))
        plot_motion_data(data, best_idx, args)

        if model_path and bin_path:
            # Export motion to MMM format.
            hypothesis_fd, hypothesis_path = mkstemp()
            with os.fdopen(hypothesis_fd, 'w') as tmp:
                if scaler:
                    decodable_hypothesis = scaler.inverse_transform(hypothesis[:, :nb_joints])
                else:
                    decodable_hypothesis = hypothesis[:, :nb_joints]
                tmp.write(get_mmm_xml_representation(decodable_hypothesis, joint_names, model_path))
            
            # Visualize motion using MMM toolkit.
            print('  visualizing hypothesis "{}" ...'.format(hypothesis_path))
            os.system(bin_path + ' --motion {} > /dev/null'.format(hypothesis_path))
            os.remove(hypothesis_path)
            print('')
        else:
            print('  MMMTools not installed, skipping motion visualization ...')
            print('')


def plot_motion_data(data, idx, args):
    nb_rows = 1
    if args.decoder == 'normal-mixture':
        nb_rows += 1
    fig, subplots = plt.subplots(nrows=nb_rows)

    # Plot prediction.
    hypothesis = data['hypotheses'][idx]
    subplots[0].pcolor(np.swapaxes(hypothesis, 0, 1))
    
    if args.decoder == 'normal-mixture':
        weights = data['predictions'][idx][:, -args.nb_mixtures - 1:-1]
        assert hypothesis.shape[0] == weights.shape[0]
        subplots[1].pcolor(np.swapaxes(weights, 0, 1), vmin=0., vmax=1.)
    plt.show()


def main(args):
    # Get the current hash of the current git commit and put it into the arguments.
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    args.git_commit = git_commit

    # Print all arguments.
    print('')
    print('Arguments:')
    args_dict = vars(args)
    sorted_keys = sorted(args_dict.keys())
    print(tabulate([(k, args_dict[k]) for k in sorted_keys]))
    print('')

    # Run the target function.
    args.func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers()

    def add_shared_arguments(p):
        p.add_argument('--batch-size', type=int, default=128)
        p.add_argument('--disable-pretty-json', action='store_true')
        p.add_argument('--decoder', choices=['normal', 'regression', 'normal-mixture'], default='normal')
        p.add_argument('--nb-mixtures', type=int, default=20)
        p.add_argument('--decoder-units', nargs='+', type=int, default=[32, 32])
        p.add_argument('--decoder-dropout', type=float, nargs=2, default=[.3, .3])
        p.add_argument('--encoder-units', nargs='+', type=int, default=[32, 32])
        p.add_argument('--encoder-dropout', type=float, nargs=2, default=[.3, .3])
        p.add_argument('--pre-units', nargs='*', type=int, default=[])
        p.add_argument('--pre-dropout', type=float, default=.3)
        p.add_argument('--pre-activation', type=str, default='relu')
        p.add_argument('--inner-units', nargs='*', type=int, default=[])
        p.add_argument('--inner-dropout', type=float, default=.3)
        p.add_argument('--inner-activation', type=str, default='relu')
        p.add_argument('--post-units', nargs='*', type=int, default=[])
        p.add_argument('--post-dropout', type=float, default=.3)
        p.add_argument('--post-activation', type=str, default='relu')
        p.add_argument('--embedding-size', type=int, default=64)
        p.add_argument('--embedding-dropout', type=float, default=0.)
        p.add_argument('--consume-less', choices=['cpu', 'mem', 'gpu'], default='gpu')
        p.add_argument('--motion-representation', choices=['abs', 'diff', 'hybrid'], default='abs')
        p.add_argument('--rnn-type', choices=['lstm', 'lstmbn', 'gru', 'gruln'], default='lstm')
        p.add_argument('--classifier-dropout', type=float, default=0.)
        p.add_argument('--classifier-l2-regularizer', type=float, default=0.)
        p.add_argument('--mixture-regularizer', type=float, default=0.)
        p.add_argument('--mixture-regularizer-type', choices=['cv', 'l2'], default=None)

        gp = p.add_mutually_exclusive_group(required=False)
        gp.add_argument('--encoder-input-peek', dest='encoder_input_peek', action='store_true')
        gp.add_argument('--no-encoder-input-peek', dest='encoder_input_peek', action='store_false')
        p.set_defaults(encoder_input_peek=False)

        gp = p.add_mutually_exclusive_group(required=False)
        gp.add_argument('--decoder-input-peek', dest='decoder_input_peek', action='store_true')
        gp.add_argument('--no-decoder-input-peek', dest='decoder_input_peek', action='store_false')
        p.set_defaults(decoder_input_peek=False)

        gp = p.add_mutually_exclusive_group(required=False)
        gp.add_argument('--decoder-context-peek', dest='decoder_context_peek', action='store_true')
        gp.add_argument('--no-decoder-context-peek', dest='decoder_context_peek', action='store_false')
        p.set_defaults(decoder_context_peek=True)

        gp = p.add_mutually_exclusive_group(required=False)
        gp.add_argument('--decoder-rnn-peek', dest='decoder_rnn_peek', action='store_true')
        gp.add_argument('--no-decoder-rnn-peek', dest='decoder_rnn_peek', action='store_false')
        p.set_defaults(decoder_rnn_peek=False)

        gp = p.add_mutually_exclusive_group(required=False)
        gp.add_argument('--batch-norm', dest='batch_norm', action='store_true')
        gp.add_argument('--no-batch-norm', dest='batch_norm', action='store_false')
        p.set_defaults(batch_norm=False)

        gp = p.add_mutually_exclusive_group(required=False)
        gp.add_argument('--bidirectional-encoder', dest='bidirectional_encoder', action='store_true')
        gp.add_argument('--no-bidirectional-encoder', dest='bidirectional_encoder', action='store_false')
        p.set_defaults(bidirectional_encoder=False)

        gp = p.add_mutually_exclusive_group(required=False)
        gp.add_argument('--motion-input-masking', dest='motion_input_masking', action='store_true')
        gp.add_argument('--no-motion-input-masking', dest='motion_input_masking', action='store_false')
        p.set_defaults(motion_input_masking=True)

        gp = p.add_mutually_exclusive_group(required=False)
        gp.add_argument('--surrogate-loss', dest='surrogate_loss', action='store_true')
        gp.add_argument('--no-surrogate-loss', dest='surrogate_loss', action='store_false')
        p.set_defaults(surrogate_loss=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('input', type=str)
    train_parser.add_argument('output', type=str)
    train_parser.add_argument('--validation-input', type=str)
    train_parser.add_argument('--nb-epoch', type=int, default=10)
    train_parser.add_argument('--optimizer', type=str, default='adam')
    train_parser.add_argument('--lr', type=float, default=.0001)
    train_parser.add_argument('--clipnorm', type=float, default=None)
    train_parser.add_argument('--clipvalue', type=float, default=None)
    mode_help = 'decides if the model is provided with the ground truth in each time step ' + \
                'regardless of the previous output (`true`) or if the output from the previous ' + \
                'epoch is used (`predict`).'
    train_parser.add_argument('--nb-annealing-epoch', type=int, default=100)
    train_parser.add_argument('--update-batch-size', type=int, default=8192)

    gp = train_parser.add_mutually_exclusive_group(required=False)
    gp.add_argument('--mask-target', dest='mask_target', action='store_true')
    gp.add_argument('--no-mask-target', dest='mask_target', action='store_false')
    train_parser.set_defaults(mask_target=False)
    gp.add_argument('--mask-target-offset', type=int, default=10)

    add_shared_arguments(train_parser)
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('input', type=str)
    evaluate_parser.add_argument('model', type=str)
    evaluate_parser.add_argument('output', type=str)
    evaluate_parser.add_argument('--validation-input', type=str)
    evaluate_parser.add_argument('--optimizer', type=str, default='adam')
    evaluate_parser.add_argument('--lr', type=float, default=.0001)
    evaluate_parser.add_argument('--clipnorm', type=float, default=None)
    evaluate_parser.add_argument('--clipvalue', type=float, default=None)
    evaluate_parser.add_argument('--nb-annealing-epoch', type=int, default=100)
    evaluate_parser.add_argument('--update-batch-size', type=int, default=8192)
    add_shared_arguments(evaluate_parser)
    evaluate_parser.set_defaults(func=evaluate)

    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('input', type=str)
    predict_parser.add_argument('model', type=str)
    predict_parser.add_argument('output', type=str)
    predict_parser.add_argument('--width', type=int, default=5)
    predict_parser.add_argument('--depth', type=int, default=250)
    predict_parser.add_argument('--include-raw-predictions', action='store_true')
    add_shared_arguments(predict_parser)
    predict_parser.set_defaults(func=predict)

    interactive_parser = subparsers.add_parser('interactive')
    interactive_parser.add_argument('input', type=str)
    interactive_parser.add_argument('model', type=str)
    interactive_parser.add_argument('--width', type=int, default=5)
    interactive_parser.add_argument('--depth', type=int, default=250)
    interactive_parser.add_argument('--maxlen', type=int, default=50)
    add_shared_arguments(interactive_parser)
    interactive_parser.set_defaults(func=interactive)
    
    main(parser.parse_args())
