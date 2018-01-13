import argparse
import os
import time
import numpy as np
import pandas as pd
import dataset

parser = argparse.ArgumentParser(description='PyTorch Amazon Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-t, --type', default='vote', type=str, metavar='TYPE',
                    help='Type of ensemble: vote, geometric, arithmetic (default: "vote"')
parser.add_argument('--multi-label', action='store_true', default=False,
                    help='multi-label target')
parser.add_argument('--tif', action='store_true', default=False,
                    help='Use tif dataset')


submission_col = ['fname', 'label']


def find_inputs(folder, filter=['results.csv']):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if rel_filename in filter:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((base, abs_filename))
    return inputs


def main():
    args = parser.parse_args()

    results = find_inputs(args.data, filter=['results.csv'])
    dfs = []
    for r in results:
        df = pd.read_csv(r[1], index_col=None)
        df = df.set_index('fname')
        dfs.append(df)

    all = pd.concat(dfs)

    id_to_label = dataset.ALL_LABELS
    id_to_label[dataset.SILENCE_INDEX] = 'silence'
    id_to_label[dataset.UNKNOWN_WORD_INDEX] = 'unknown'  # this mapping used for output

    sum = all.groupby(['fname']).sum()
    sum /= len(dfs)
    sum_arr = sum.as_matrix()
    fnames = sum.index.values
    idx = np.argmax(sum_arr, axis=1)
    log_probs = sum_arr[np.arange(len(idx)), idx]
    probs = np.exp(log_probs)
    labels = np.array(id_to_label)[idx]
    ensemble = pd.DataFrame(
        data={'fname': fnames, 'label': labels, 'prob': probs}, columns=['fname', 'label', 'prob'])
    ensemble.to_csv('./ensemble.csv', index=False)


if __name__ == '__main__':
    main()