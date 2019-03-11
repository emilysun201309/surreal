import argparse

import matplotlib.pyplot as plt
import numpy as np
import h5py



def plot_data(ax, data, title=None, args=None):
    if args.start_idx:
        data = data[args.start_idx:, ...]
    heatmap = ax.pcolor(np.swapaxes(data, 0, 1), vmin=-1., vmax=1.)
    if not args.plot_both:
        plt.colorbar(heatmap)

    # Configure y axis.
    yticks = np.arange(1, data.shape[1] + 1).tolist()
    ylabels = [str(tick) if (tick - 1) % 5 == 0 else '' for tick in yticks]
    ax.set_yticks(yticks, ylabels)
    #ax.set_ylim([0, data.shape[1]])
    ax.set_ylabel('feature dimension')

    # Configure x axis.
    ax.set_xlabel('timestep')
    ax.set_xlim([0, data.shape[0]])


def decode_tokens(tokens, vocabulary):
    return ' '.join([vocabulary[token] for token in tokens])


def insert_newlines(string, every=64):
    lines = []
    for i in xrange(0, len(string), every):
        lines.append(string[i:i+every])
    return '\n'.join(lines)


def main(args):
    f = h5py.File(args.input, 'r')

    if args.idx is None:
        # Select a random element.
        nb_motions = len(f['motion_inputs'])
        sample_idx = np.random.randint(nb_motions)
    else:
        sample_idx = args.idx

    # Find associated annotations.
    annotation_indexes = []
    mat_id = None
    for motion_idx, annotation_idx, id_idx in f['mapping']:
        if motion_idx == sample_idx:
            mat_id = f['ids'][id_idx]
            annotation_indexes.append(annotation_idx)
    assert id_idx is not None
    
    # Get data.
    motion_input = f['motion_inputs'][sample_idx]
    motion_target = f['motion_targets'][sample_idx]
    annotation_inputs = f['annotation_inputs'][annotation_indexes]
    annotation_targets = f['annotation_targets'][annotation_indexes]
    vocabulary = f['vocabulary']

    print('Visualizing index {} from "{}" with ID {}:'.format(sample_idx, args.input, mat_id))
    print('')
    
    print('All {} annotation inputs:'.format(len(annotation_inputs)))
    for idx, annotation_input in enumerate(annotation_inputs):
        print('  {} decoded: {}'.format(idx + 1, decode_tokens(annotation_input, vocabulary)))
        print('  {} indexes: {}'.format(idx + 1, ','.join([str(x) for x in annotation_input])))
        print('')
    print('')

    print('All {} annotation targets:'.format(len(annotation_targets)))
    for idx, annotation_target in enumerate(annotation_targets):
        print('  {} decoded: {}'.format(idx + 1, decode_tokens(annotation_target, vocabulary)))
        print('  {} indexes: {}'.format(idx + 1, ','.join([str(x) for x in annotation_target])))
        print('')
    print('')

    # Visualize motion data.
    print('Visualizing motion input and motion target ...')
    if args.plot_both:
        fig, subplots = plt.subplots(nrows=2, figsize=args.figsize)
    else:
        fig, subplots = plt.subplots(nrows=1, figsize=args.figsize)
        subplots = [subplots]
    for data, plot in zip([motion_input, motion_target], subplots):
        print(data[0, :], data[-1, :])
        plot_data(plot, data, args=args)
    plt.tight_layout(pad=1.)
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
    print('done')

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument('--plot-both', type=int, default=1)
    parser.add_argument('--start-idx', type=int, default=None)
    parser.add_argument('--figsize', nargs=2, type=float, default=(20, 10))
    parser.add_argument('--output', type=str, default=None)
    main(parser.parse_args())
