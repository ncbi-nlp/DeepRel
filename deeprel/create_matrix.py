"""
Usage:
    create_matrix.py [options] --vocab=<file> --all=<directory> --output=<file> INPUT_FILE
    
Options:
    --verbose
    --vocab=<file>
    --all=<directory>
    --output=<file>
"""

import collections
import json
import logging
import sys

import os
from pathlib import Path

import docopt
import numpy as np

from cli_utils import parse_args
from deeprel.model import re_vocabulary
from deeprel import utils


class Transformer(object):
    def __init__(self, vocabs):
        """
        Args:
            vocabs(ReVocabulary)
        """
        self.vocabs = vocabs

    def transform(self, toks, label):
        pass


class NoDepTransformer(Transformer):
    def __init__(self, vocabs):
        super(NoDepTransformer, self).__init__(vocabs)

    def transform(self, toks, label):
        """Transform toks to indices.

        Return:
            list: a list of 6 elements
            int: label
            list: token['word']
        """
        x = []
        x_text = []  # for debug

        for i, tok in enumerate(toks):
            tokx = []
            for key in self.vocabs.keys:
                if key in tok:
                    tokx.append(self.vocabs[key].get(str(tok[key])))
            x_text.append((i, [tok['word']]))
            x.append(tokx)

        return self.pad(self.vocabs.max_len, len(self.vocabs.keys), x), \
               self.vocabs.label_vocab.get(label), \
               x_text

    @classmethod
    def pad(cls, row, height, x):
        x_array = np.zeros(shape=(row, height), dtype=np.int64)
        for idx, elem in enumerate(x):
            for idt, k in enumerate(elem):
                x_array[idx, idt] = k
        return x_array


class MatrixCreater(object):
    def __init__(self, vocabs_filename, json_dir):
        self.vocabs = re_vocabulary.load(vocabs_filename)
        self.json_dir = json_dir
        self.transformer = NoDepTransformer(self.vocabs)

    def create_matrix(self, src, dst):
        xs = []
        ys = []
        xs_text = []

        for obj, ex in utils.example_iterator([src], Path(self.json_dir)):
            if 'toks' in ex and 'shortest path' in ex:
                x, y, x_text = self.transformer.transform(ex['toks'], ex['label'])
                xs.append(x)
                ys.append(y)
                xs_text.append((ex['id'], x_text))
        self.save(xs, ys, xs_text, dst)

    def save(self, xs, ys, xs_text, dst):
        x = np.stack(xs, axis=0)
        y = np.stack(ys, axis=0)
        one_hot = np.zeros((y.size, len(self.vocabs.label_vocab)))
        one_hot[np.arange(y.size), y] = 1

        logging.debug('x shape: %s', x.shape)
        logging.debug('y shape: %s', one_hot.shape)
        np.savez(dst, x=x, y=one_hot)

        with open(dst + '.txt', 'w') as f:
            for docid, x_text in xs_text:
                s = docid + '\n'
                for idx, k in x_text:
                    s += '{} = {}\n'.format(idx, ' <-- '.join(k))
                f.write(s)


def read_matrix(src):
    logging.info('Read matrix: %s', src)
    npzfile = np.load(src)
    return npzfile['x'], npzfile['y']


if __name__ == '__main__':
    argv = parse_args(__doc__)
    mc = MatrixCreater(argv['--vocab'], argv['--all'])
    mc.create_matrix(argv['INPUT_FILE'], argv['--output'])