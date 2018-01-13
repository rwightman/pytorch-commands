import os
import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    'input',
    type=str,
    metavar='FILE',
    help='path to input csv')
parser.add_argument(
    '-o', '--output',
    type=str,
    metavar='FILE',
    default='./submission-filtered.csv',
    help='path to out csv')
parser.add_argument(
    '--words',
    type=str,
    default='yes,no,up,down,left,right,on,off,stop,go'
            #'zero,one,two,three,four,five,six,seven,eight,nine,'
            #'bird,dog,cat,bed,house,tree,marvin,sheila,happy,wow'
    )
parser.add_argument(
    '-t', '--thresh',
    type=float,
    default=0.3)


def main():
    args = parser.parse_args()

    results = pd.read_csv(args.input)
    total = len(results.index)
    g = results.groupby(['label'])
    for i, c in g['fname'].count().iteritems():
        print("%s\t%d\t%f" % (i, c, 100 * c / total))

    accept = set(args.words.split(','))
    accept.add('silence')
    print(accept)

    results['label'] = results['label'].map(
        lambda x: x if x in accept else 'unknown')
    results['label'] = results.apply(
        lambda x: x['label'] if x['prob'] >= args.thresh else 'silence', axis=1)
    results = results[['fname', 'label']]
    results.to_csv(args.output, index=False)

    g = results.groupby(['label'])
    for i, c in g['fname'].count().iteritems():
        print("%s\t%d\t%f" % (i, c, 100 * c / total))

if __name__ == '__main__':
    main()

