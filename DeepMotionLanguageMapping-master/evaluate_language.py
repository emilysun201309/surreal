import argparse
import json
import random
from tempfile import mktemp
import os

import numpy as np


def compute_sentence_bleu_scores(prediction):
    references = [reference.split() for reference in prediction['references']]
    scores = []

    # Go through hypothesis as sorted by their probabilities. Usually, the predictions
    # are already sorted, but better be safe than sorry.
    for idx in np.argsort(prediction['log_probabilities'])[::-1]:
        hypothesis = prediction['hypotheses'][idx]
        try:
            score = sentence_bleu(references, hypothesis.split())
        except ZeroDivisionError:
            score = 0.
        scores.append(score)
    return scores


def main(args):
    # Load all predictions.
    print('Loading predictions "{}" ...'.format(args.input))
    nb_max_references = 0
    with open(args.input, 'r') as f:
        ps = json.load(f)

        # Filter out duplicates.
        used_ids = []
        predictions = []
        for p in ps:
            if p['id'] in used_ids:
                continue
            used_ids.append(p['id'])
            predictions.append(p)
            nb_max_references = max(nb_max_references, len(p['references']))
    print('done, {} predictions, nb_max_references={}'.format(len(predictions), nb_max_references))
    print('')

    hypotheses = []
    references = [[] for _ in range(nb_max_references)]
    for pred in predictions:
        if args.mode == 'baseline' and len(pred['references']) < 2:
            continue

        sorted_idx = list(reversed(np.argsort(pred['log_probabilities'])))[args.hypothesis_idx]
        if args.mode == 'baseline':
            nb_references = len(pred['references'])
            ref_idx = random.choice(range(nb_references))
            hypotheses.append(pred['references'][ref_idx])
            del pred['references'][ref_idx]
            assert len(pred['references']) == nb_references - 1
            assert len(pred['references']) > 0
        else:
            hypotheses.append(pred['hypotheses'][sorted_idx])
        for idx in range(nb_max_references):
            ref_idx = min(idx, len(pred['references']) - 1)
            references[idx].append(pred['references'][ref_idx])
    assert len(references) == nb_max_references
    assert np.all(np.array([len(rs) for rs in references]) == len(hypotheses))

    path_hypotheses = mktemp()
    path_references = mktemp()
    with open(path_hypotheses, 'w') as f:
        f.write('\n'.join(hypotheses))
    for idx, rs in enumerate(references):
        with open(path_references + str(idx), 'w') as f:
            f.write('\n'.join(rs))
    print(os.system('perl vendor/multi-bleu.perl {} < {}'.format(path_references, path_hypotheses)))
    os.remove(path_hypotheses)
    for idx in range(nb_max_references):
        os.remove(path_references + str(idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--mode', type=str, choices=['corpus', 'baseline'], default='corpus')
    parser.add_argument('--hypothesis-idx', type=int, default=0)
    main(parser.parse_args())
