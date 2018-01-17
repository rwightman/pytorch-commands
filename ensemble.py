import argparse
import os
import time
import numpy as np
import pandas as pd
import dataset

parser = argparse.ArgumentParser(description='Ensembler')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')


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

    id_to_label = dataset.get_labels()
    results = find_inputs(args.data, filter=['results.csv'])
    dfs = []
    for r in results:
        df = pd.read_csv(r[1], index_col=None)
        df = df.set_index('fname')
        dfs.append(df)
    all = pd.concat(dfs)

    sum = all.groupby(['fname']).sum()
    sum /= len(dfs)
    sum_arr = sum.as_matrix()
    fnames = sum.index.values
    probs_arr = np.exp(sum_arr)
    #probs_arr[:, 0] += 0.3
    #probs_arr[:, 1] += 0.16
    idx = np.argmax(probs_arr, axis=1)
    probs = probs_arr[np.arange(len(idx)), idx]
    labels = np.array(id_to_label)[idx]
    ensemble = pd.DataFrame(
        data={'fname': fnames, 'label': labels, 'prob': probs}, columns=['fname', 'label', 'prob'])
    ensemble.to_csv('./ensemble.csv', index=False)


if __name__ == '__main__':
    main()