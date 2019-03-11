import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    with open(args.input, 'r') as f:
        history = json.load(f)
    if args.metric not in history:
        exit('History "{}" does not contain the metric "{}". Possible values are: {}'.format(args.input, args.metric, history.keys()))
    if 'nb_epoch' in history:
        epoch = np.array(range(history['nb_epoch'])) + 1
    else:
        epoch = np.array(history['epochs']) + 1
    metric_train = np.array(history[args.metric])
    metric_valid = np.array(history['val_' + args.metric])
    assert epoch.shape == metric_train.shape
    assert epoch.shape == metric_valid.shape

    plt.plot(epoch, metric_train, label='training', linewidth=2.0)
    plt.plot(epoch, metric_valid, label='validation', linewidth=2.0)
    plt.xlabel('epoch')
    plt.ylabel(args.metric)
    plt.legend()
    plt.tight_layout(pad=0.)

    if args.output:
        print('Saving plot to "{}" ...'.format(args.output))
        plt.savefig(args.output)
        print('done')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--metric', type=str, default='loss')
    main(parser.parse_args())
