from __future__ import print_function

import os
import logging
import json
import collections

from pynih.machine_learning import metrics


def add(gld, easyhard, g, label):
    gold_row = None
    for toks in gld:
        if toks[0] == label \
                and ((g[1].endswith(toks[2]) and g[2].endswith(toks[3]))
                     or (g[1].endswith(toks[3]) and g[2].endswith(toks[2]))):
            gold_row = toks
            break
    if not gold_row:
        return None

    for toks in easyhard:
        if toks[0] == gold_row[1]:
            gold_row.append(toks[-1])
            return gold_row

    return None


def read(rstfile, gldfile, fold):
    y_test = []
    y_pred = []
    with open(rstfile) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split('\t')
            y_test.append(int(toks[0]))
            y_pred.append(int(toks[1]))

    y_gold = []
    with open(gldfile) as f:
        for line in f:
            line = line.strip()
            if not line or (line[0] != 'A' and line[0] != 'B'):
                continue
            toks = line.split(',')
            y_gold.append(toks)

    assert len(y_test) == len(y_gold), 'fold {}: {} vs {}'.format(fold, len(y_test), len(y_gold))

    return y_test, y_pred, y_gold


def read_fold(data_dir, learning_format, easyhard):
    total_text = ''
    for fold in range(0, 10):
        rstfile = os.path.join(data_dir, 'cnn_model{0}/TestSet_{0}-matrix.rst'.format(fold))
        gldfile = os.path.join(data_dir, 'cnn_model/TestSet_{0}-matrix.npz.txt'.format(fold))

        y_test, y_pred, y_gold = read(rstfile, gldfile, fold)

        tp = 0
        fp = 0
        fn = 0
        for g, t, p in zip(y_gold, y_test, y_pred):
            if t == p == 1:
                tp += 1
                # total_text += str(add(learning_format, easyhard, g, 'True')) + '\n'
                # break
            if t == 1 != p:
                fn += 1
            if t == 0 != p:
                fp += 1
            if t == p == 0:
                total_text += str(add(learning_format, easyhard, g, 'False')) + '\n'
    return total_text


def main():
    datadir = os.path.expanduser('~/data/bionlp2017/')
    learning_format = []
    with open(os.path.join(datadir, 'BioInfer-learning-format.tsv')) as fp:
        for line in fp:
            learning_format.append(line.strip().split())
    with open(os.path.join(datadir, 'AImed-learning-format.tsv')) as fp:
        for line in fp:
            learning_format.append(line.strip().split())

    easyhard = []
    with open(os.path.join(datadir, 'pairdifficulty.csv')) as fp:
        for line in fp:
            easyhard.append(line.strip().split(','))

    total_text = ''
    total_text += read_fold(os.path.expanduser('~/data/deeprel/aimed-win3'), learning_format, easyhard)
    total_text += read_fold(os.path.expanduser('~/data/deeprel/bioinfer'), learning_format, easyhard)

    with open(os.path.expanduser('~/data/bionlp2017/tn.txt'), 'w') as fp:
        fp.write(total_text)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()

