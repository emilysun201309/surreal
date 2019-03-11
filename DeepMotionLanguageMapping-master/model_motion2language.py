import argparse
import os
import cPickle as pickle
import json
import timeit
import subprocess

import h5py
import numpy as np; np.random.seed(42)
from tabulate import tabulate

from keras.layers import (Input, LSTM, GRU, Dense, merge, TimeDistributed, RepeatVector, Activation,
    Embedding, Bidirectional, Dropout, Masking)
from keras.models import Model
from keras.utils.generic_utils import Progbar
from keras.callbacks import History, ModelCheckpoint
from keras.utils.visualize_util import plot
from keras.optimizers import get as get_optimizer
from keras.regularizers import l2
import keras.backend as K
if K.backend() != 'theano':
    raise RuntimeError('only supports Theano backend')

from sklearn.preprocessing import OneHotEncoder


def load_data_predict(path):
    f = h5py.File(path, 'r')
    motions = f['motion_inputs']
    print motions.shape, f['ids'].shape
    annotations = f['annotation_targets']
    mapping = f['mapping']
    vocabulary = f['vocabulary']
    nb_vocabulary = len(vocabulary)

    # Collect all annotation indexes for a given motion index.
    annotation_indexes_for_motion_idx = {}
    id_indexes_for_motion_idx = {}
    for motion_idx, annotation_idx, id_idx in mapping:
        if motion_idx not in annotation_indexes_for_motion_idx:
            annotation_indexes_for_motion_idx[motion_idx] = []
        annotation_indexes_for_motion_idx[motion_idx].append(annotation_idx)

        id_indexes_for_motion_idx[motion_idx] = id_idx
    assert len(annotation_indexes_for_motion_idx) == len(motions)

    # Now, load the actual motion data and the corresponding language references.
    X_motion = []
    references = []
    ids = []
    f_ids = list(f['ids'])
    for motion_idx, annotation_indexes in annotation_indexes_for_motion_idx.iteritems():
        X_motion.append(motions[motion_idx])
        references.append([annotations[annotation_idx] for annotation_idx in annotation_indexes])

        id_idx = id_indexes_for_motion_idx[motion_idx]
        ids.append(f_ids[id_idx])
    assert len(X_motion) == len(references)
    X_motion = np.array(X_motion).astype('float32')

    # We're done. We do not encode the references, since we won't use them in the model.
    # They are only used to provide the ground truth for later evaluation purposes.
    assert len(ids) == len(X_motion)
    return X_motion, references, vocabulary, ids


def load_data_train(path):
    f = h5py.File(path, 'r')
    motions = f['motion_inputs']
    annotations = f['annotation_targets']
    mapping = f['mapping']
    vocabulary = f['vocabulary']
    start_symbol = f['vocabulary'].attrs['start_symbol']
    start_idx = list(vocabulary).index(start_symbol)
    nb_vocabulary = len(vocabulary)

    # Create usable data for training.
    X_motion = []
    language = []
    for motion_idx, annotation_idx, _ in mapping:
        X_motion.append(motions[motion_idx])
        language.append(annotations[annotation_idx])
    assert len(X_motion) == len(language)
    X_motion = np.array(X_motion).astype('float32')

    # Move language one back, since this is the previous time step.
    X_language = np.array(language).astype('int32')[:, :-1]
    X_language = np.hstack([np.ones((X_language.shape[0], 1), dtype='int32') * start_idx, X_language])

    # Encode targets as probabilities.
    encoder = OneHotEncoder(n_values=nb_vocabulary)
    encoder.fit(language)
    Y = encoder.transform(language).toarray().astype(bool).reshape((X_motion.shape[0], annotations.shape[1], nb_vocabulary))

    return X_motion, X_language, Y, vocabulary


def load_weights(path, encoder, decoder):
    flattened_layers = encoder.layers + decoder.layers
    names = list(set([l.name for l in flattened_layers]))
    if len(names) != len(flattened_layers):
        raise Exception('The layers of the encoder and decoder contain layers with the same name. Please use unique names.')
    f = h5py.File(path, 'r')
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    weight_value_tuples = []
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = None
            for l in flattened_layers:
                if l.name == name:
                    layer = l
                    break
            if layer is None:
                raise Exception('The layer "{}", for which we found weights, does not exist'.format(name))
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
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
            stateful=stateful, name=name, gamma_init=args.gruln_gamma_init)
    else:
        raise RuntimeError('Unknown RNN type "{}".'.format(args.rnn_type))
    return layer


def build_encoder(x, args):
    dropout_W = args.encoder_dropout[0]
    dropout_U = args.encoder_dropout[1]
    depth = len(args.encoder_units)

    layers = []
    for idx, nb_units in enumerate(args.encoder_units):
        # Merge previous output with (transformed) input if peeking is enabled.
        input_layer = None
        if idx == 0:
            input_layer = x
        elif args.encoder_input_peek:
            input_layer = merge([layers[-1], x], mode='concat')
        else:
            input_layer = layers[-1]
        assert input_layer is not None

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
        encoder = encoder_layer(input_layer)
        layers.append(encoder)
    return layers[-1]


def build_decoder(context, previous_timestep, nb_vocabulary, args, stateful=False):
    # Create the stacked LSTM decoder.
    dropout_W = args.decoder_dropout[0]
    dropout_U = args.decoder_dropout[1]
    depth = len(args.decoder_units)

    layers = []
    rnn_layers = []
    for idx, nb_units in enumerate(args.decoder_units):
        # Define the input from the previous layer(s).
        inputs = [layers[-1]] if len(layers) > 0 else []
        if idx == 0 or args.decoder_context_peek:
            inputs.append(context)
        if idx == 0 or args.decoder_input_peek:
            inputs.append(previous_timestep)
        input_layer = None
        if len(inputs) > 1:
            input_layer = merge(inputs, mode='concat')
        else:
            input_layer = inputs[0]
        assert input_layer is not None

        name = 'decoder_{}'.format(idx)
        decoder = get_rnn_layer(nb_units, return_sequences=True, stateful=stateful, name=name,
            dropout_W=dropout_W, dropout_U=dropout_U, args=args)(input_layer)
        layers.append(decoder)
        rnn_layers.append(decoder)

    if args.decoder_rnn_peek:
        layers.append(merge(rnn_layers, mode='concat'))

    # Finally, add a simple softmax classifier on top, distributed over time.
    if args.classifier_dropout > 0.:
        layers.append(Dropout(args.classifier_dropout)(layers[-1]))
    classifier = TimeDistributed(Dense(nb_vocabulary,
        W_regularizer=l2(args.classifier_l2_regularizer)))(layers[-1])
    return Activation('softmax')(classifier)


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


def prepare_for_training(output_path, args):
    # Load training data.
    print('Loading training data "{}" ...'.format(args.input))
    X_motion_train, X_language_train, Y_train, vocabulary = load_data_train(args.input)
    nb_vocabulary = len(vocabulary)
    print('done, X_motion_train = {}, X_language_train = {}, Y_train = {}, nb_vocabulary = {}'.format(X_motion_train.shape, X_language_train.shape, Y_train.shape, nb_vocabulary))
    print('')

    # Load validation data, if applicable.
    valid_data = None
    if args.validation_input:
        print('Loading validation data "{}" ...'.format(args.validation_input))
        X_motion_valid, X_language_valid, Y_valid, v = load_data_train(args.validation_input)
        assert np.all(vocabulary[:] == v[:])
        print('done, X_motion_valid = {}, X_language_valid = {}, Y_valid = {}, nb_vocabulary = {}'.format(X_motion_valid.shape, X_language_valid.shape, Y_valid.shape, nb_vocabulary))
        print('')
        valid_data = ([X_motion_valid, X_language_valid], Y_valid)

    # The input are motion sequences of shape (nb_samples, nb_steps, nb_dim).
    motion_input = Input(shape=X_motion_train.shape[1:], name='motion_input')
    masked_motion_input = Masking()(motion_input)

    # The other input to the model is the previous language output. During training, we set this
    # to the ground truth.
    language_input = Input(shape=(X_language_train.shape[1],), dtype='int32', name='language_input')
    embedded_language = Embedding(nb_vocabulary, args.embedding_size, input_length=X_language_train.shape[1],
        name='embedding_language', dropout=args.embedding_dropout, mask_zero=True)(language_input)

    # Create the encoder and decoder.
    encoder = build_encoder(masked_motion_input, args)
    repeated_context = RepeatVector(Y_train.shape[1])(encoder)
    decoder = build_decoder(repeated_context, embedded_language, nb_vocabulary, args)

    # Create the model.
    print('Compiling the model ...')
    model = Model(input=[motion_input, language_input], output=decoder)
    print(model.summary())
    optimizer = get_optimizer(args.optimizer)
    K.set_value(optimizer.lr, args.lr)
    if args.clipnorm is not None:
        optimizer.clipnorm = args.clipnorm
    if args.clipvalue is not None:
        optimizer.clipvalue = args.clipvalue
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    print('done')
    print('')

    data_train = (X_language_train, X_motion_train, Y_train, nb_vocabulary)
    return data_train, valid_data, model, optimizer


def train(args):
    output_path = prepare_output(args)
    train_data, valid_data, model, optimizer = prepare_for_training(output_path, args)
    X_language_train, X_motion_train, Y_train, nb_vocabulary = train_data

    # Save model information to output.
    print('Saving model information to "{}" ...'.format(output_path))
    with open(os.path.join(output_path, 'model.json'), 'w') as f:
        f.write(model.to_json())
    plot(model, to_file=os.path.join(output_path, 'model.pdf'), show_shapes=True)
    plot(model, to_file=os.path.join(output_path, 'model.dot'), show_shapes=True)
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
        model.fit([X_motion_train, X_language_train], Y_train, batch_size=args.batch_size,
            nb_epoch=args.nb_epoch, validation_data=valid_data, callbacks=[history, checkpoint])
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
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
    train_data, valid_data, model, optimizer = prepare_for_training(output_path, args)
    X_language_train, X_motion_train, Y_train, nb_vocabulary = train_data

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
            metrics = model.evaluate([X_motion_train, X_language_train], Y_train, batch_size=args.batch_size)
            train_metrics.append(metrics if isinstance(metrics, (list, tuple)) else [metrics])
            print(zip(model.metrics_names, train_metrics[-1]))
            if valid_data:
                X_language_valid, X_motion_valid, Y_valid = valid_data
                metrics = model.evaluate([X_motion_valid, X_language_valid], Y_valid, batch_size=args.batch_size)
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


def perform_random_sampling(preds, decoder, end_idx, previous_tokens, log_probabilities, done):
    assert preds.ndim == 2
    width, nb_vocabulary = preds.shape

    # Sample.
    token_indexes = []
    probas = []
    for pred in preds:
        token_idx = np.random.choice(range(nb_vocabulary), p=pred)
        token_indexes.append(token_idx)
        probas.append(pred[token_idx])
    
    # Book-keeping.
    for idx, (token_idx, proba) in enumerate(zip(token_indexes, probas)):
        if done[idx]:
            continue
        
        previous_tokens[idx].append(token_idx)
        log_probabilities[idx] += np.log(proba)
        if token_idx == end_idx:
            done[idx] = True

    return previous_tokens, log_probabilities, done


def perform_beam_search(preds, decoder, end_idx, previous_tokens, log_probabilities, done):
    assert preds.ndim == 2
    width, nb_vocabulary = preds.shape
    
    # Compute the total probabilities up to this point.
    combined_log_probas = []
    for previous_log_proba, current_log_probas in zip(log_probabilities, np.log(preds)):
        combined_log_probas.append(previous_log_proba + current_log_probas)
    combined_log_probas = np.array(combined_log_probas)
    assert combined_log_probas.shape == (width, nb_vocabulary)

    # Next, find the most likely sequences.
    sorted_indexes = np.argsort(combined_log_probas.flatten())[::-1]
    assert sorted_indexes.shape == (width * nb_vocabulary,)

    # Get the state of the decoder. We need to re-arrange the state since we potentially expand
    # the same path in two different directions (state gets duplicated) or we truncate a previously
    # expanded path (state is replaced).
    stateful_layers = [layer for layer in decoder.layers if getattr(layer, 'stateful', False)]
    old_all_states = [K.batch_get_value(layer.states) for layer in stateful_layers]
    new_all_states = [[np.zeros(state.shape) for state in states] for states in old_all_states]

    # Only keep following most promising paths. However, we want to maintain some diversity,
    # so ensure that we do not follow the same exact path more than once.
    old_log_probabilities = log_probabilities[:]
    old_previous_tokens = previous_tokens[:]
    old_done = done[:]
    next_idx = 0
    for sorted_idx in sorted_indexes:
        if next_idx >= width:
            break

        beam_idx = sorted_idx / nb_vocabulary
        assert 0 <= beam_idx < width
        token_idx = sorted_idx % nb_vocabulary
        assert 0 <= token_idx < nb_vocabulary

        if next_idx > 0 and previous_tokens[next_idx - 1] == old_previous_tokens[beam_idx][:] + [token_idx]:
            # This sequence is not novel, skip it.
            continue

        # Copy over previous state.
        done[next_idx] = old_done[beam_idx]
        previous_tokens[next_idx] = old_previous_tokens[beam_idx][:]
        log_probabilities[next_idx] = old_log_probabilities[beam_idx]
        for new_states, old_states in zip(new_all_states, old_all_states):
            for new_state, old_state in zip(new_states, old_states):
                new_state[next_idx] = np.copy(old_state[beam_idx])
        if done[next_idx]:
            # This path has already reached a final state, so we're done here.
            next_idx += 1
            continue

        # Expand.
        previous_tokens[next_idx] += [token_idx]
        log_probabilities[next_idx] += np.log(preds[beam_idx, token_idx])
        if token_idx == end_idx:
            done[next_idx] = True
        next_idx += 1

    # Apply new states.
    for layer, states in zip(stateful_layers, new_all_states):
        K.batch_set_value(zip(layer.states, states))

    return previous_tokens, log_probabilities, done


def decode(context, decoder, vocabulary, references, id, args):
    nb_vocabulary = len(vocabulary)

    vocab_list = list(vocabulary)
    end_idx = vocab_list.index(vocabulary.attrs['end_symbol'])
    start_idx = vocab_list.index(vocabulary.attrs['start_symbol'])
    padding_idx = vocab_list.index(vocabulary.attrs['padding_symbol'])

    # Prepare data structures for graph search.
    previous_tokens = [[start_idx] for _ in xrange(args.width)]
    repeated_context = np.repeat(context.reshape(1, context.shape[-1]), args.width, axis=0)
    repeated_context = repeated_context.reshape(args.width, 1, context.shape[-1])
    log_probabilities = [0. for _ in xrange(args.width)]
    done = [False for _ in xrange(args.width)]

    # Reset the decoder.
    decoder.reset_states()

    # Iterate over time.
    for _ in xrange(args.depth):
        previous_timestep = np.array([tokens[-1] for tokens in previous_tokens], dtype='int32')
        preds = decoder.predict_on_batch([repeated_context, previous_timestep])
        preds = preds.reshape(args.width, nb_vocabulary)

        # Perform actual decoding.
        if args.decoder == 'beam':
            fn = perform_beam_search
        elif args.decoder == 'random':
            fn = perform_random_sampling
        else:
            fn = None
            raise ValueError('Unknown decoder "{}"'.format(args.decoder))
        previous_tokens, log_probabilities, done = fn(preds, decoder, end_idx, previous_tokens, log_probabilities, done)

        # Check if we're done before reaching `args.depth`.
        if np.all(done):
            break

    # Convert indexes to human-readable sentences.
    hypotheses = []
    for tokens in previous_tokens:
        # We ignore the first token since this is always the GO token.
        sentence = ' '.join([vocabulary[token] for token in tokens[1:]])
        hypotheses.append(sentence)
    human_readable_references = []
    for tokens in references:
        sentence = ' '.join([vocabulary[token] for token in tokens if token != padding_idx])
        human_readable_references.append(sentence)

    # Record data.
    data = {
        'hypotheses': hypotheses,
        'log_probabilities': log_probabilities,
        'references': human_readable_references,
        'id': id,
    }
    return data


def predict(args):
    output_path = prepare_output(args)

    # Load data.
    print('Loading data "{}" ...'.format(args.input))
    X, references, vocabulary, ids = load_data_predict(args.input)
    nb_vocabulary = len(vocabulary)
    print('done, X = {}, references = {}, nb_vocabulary = {}'.format(X.shape, len(references), nb_vocabulary))
    print('')

    # The input are motion sequences of shape (nb_samples, nb_steps, nb_dim).
    motion_input = Input(shape=X.shape[1:], name='motion_input')
    masked_motion_input = Masking()(motion_input)
    encoder = build_encoder(masked_motion_input, args)
    model_encoder = Model(input=motion_input, output=encoder)

    # Next, decode the sequences.
    context_input = Input(batch_shape=(args.width, 1, encoder._keras_shape[-1]), name='context_input')
    previous_input = Input(batch_shape=(args.width, 1), dtype='int32', name='previous_input')
    embedded_previous_input = Embedding(nb_vocabulary, args.embedding_size, input_length=1,
        name='embedding_language', dropout=args.embedding_dropout, mask_zero=True)(previous_input)
    decoder = build_decoder(context_input, embedded_previous_input, nb_vocabulary, args, stateful=True)
    model_decoder = Model(input=[context_input, previous_input], output=decoder)

    # Compile both models. We don't use the optimizer, so we specify anything here.
    print('Compiling models and loading weights "{}" ...'.format(args.model))
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

    # Decode sentences.
    print('Decoding {} context vectors using "{}" ...'.format(contexts.shape[0], args.decoder))
    decoded_data = []
    progbar = Progbar(target=contexts.shape[0])
    try:
        for sample_idx, context in enumerate(contexts):
            data = decode(context, decoder=model_decoder, vocabulary=vocabulary,
                id=ids[sample_idx], references=references[sample_idx], args=args)
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
    with open(os.path.join(output_path, 'predict.json'), 'w') as f:
        dump_data(decoded_data, f, args)
    with open(os.path.join(output_path, 'predict_args.json'), 'w') as f:
        dump_data(serializable_args(args), f, args)
    print('done')


def main(args):
    # Get the current hash of the current git commit and put it into the arguments.
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    args.git_commit = git_commit

    # Print all arguments.
    print('')
    print('Arguments:')
    print(tabulate(vars(args).iteritems()))
    print('')

    # Run the target function.
    args.func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers()

    def add_shared_arguments(p):
        p.add_argument('--disable-pretty-json', action='store_true')
        p.add_argument('--batch-size', type=int, default=128)
        p.add_argument('--decoder-units', nargs='+', type=int, default=[128, 128])
        p.add_argument('--decoder-dropout', type=float, nargs=2, default=[.2, .2])
        p.add_argument('--encoder-units', nargs='+', type=int, default=[128, 128])
        p.add_argument('--encoder-dropout', type=float, nargs=2, default=[.2, .2])
        p.add_argument('--embedding-size', type=int, default=64)
        p.add_argument('--embedding-dropout', type=float, default=0.)
        p.add_argument('--classifier-dropout', type=float, default=0.)
        p.add_argument('--classifier-l2-regularizer', type=float, default=0.)
        p.add_argument('--consume-less', choices=['cpu', 'mem', 'gpu'], default='gpu')
        p.add_argument('--rnn-type', choices=['lstm', 'lstmbn', 'gru', 'gruln'], default='lstm')
        p.add_argument('--gruln-gamma-init', type=float, default=1.)

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
        gp.add_argument('--bidirectional-encoder', dest='bidirectional_encoder', action='store_true')
        gp.add_argument('--no-bidirectional-encoder', dest='bidirectional_encoder', action='store_false')
        p.set_defaults(bidirectional_encoder=False)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('input', type=str)
    train_parser.add_argument('output', type=str)
    train_parser.add_argument('--validation-input', type=str)
    train_parser.add_argument('--nb-epoch', type=int, default=10)
    train_parser.add_argument('--optimizer', type=str, default='adam')
    train_parser.add_argument('--lr', type=float, default=.001)
    train_parser.add_argument('--clipnorm', type=float, default=None)
    train_parser.add_argument('--clipvalue', type=float, default=None)
    add_shared_arguments(train_parser)
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('input', type=str)
    evaluate_parser.add_argument('model', type=str)
    evaluate_parser.add_argument('output', type=str)
    evaluate_parser.add_argument('--validation-input', type=str)
    evaluate_parser.add_argument('--nb-epoch', type=int, default=10)
    evaluate_parser.add_argument('--optimizer', type=str, default='adam')
    evaluate_parser.add_argument('--lr', type=float, default=.001)
    evaluate_parser.add_argument('--clipnorm', type=float, default=None)
    evaluate_parser.add_argument('--clipvalue', type=float, default=None)
    add_shared_arguments(evaluate_parser)
    evaluate_parser.set_defaults(func=evaluate)

    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('input', type=str)
    predict_parser.add_argument('model', type=str)
    predict_parser.add_argument('output', type=str)
    predict_parser.add_argument('--width', type=int, default=5)
    predict_parser.add_argument('--depth', type=int, default=50)
    predict_parser.add_argument('--decoder', choices=['beam', 'random'], default='beam')
    add_shared_arguments(predict_parser)
    predict_parser.set_defaults(func=predict)
    
    main(parser.parse_args())
