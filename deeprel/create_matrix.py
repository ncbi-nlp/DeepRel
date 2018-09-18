"""
Usage:
    create_matrix.py matrix [options] --vocab=<file> --all=<directory> --output=<file> INPUT_FILE
    create_matrix.py sp [options] --vocab=<file> --all=<directory> --output=<file> INPUT_FILE
    
Options:
    --verbose
    --vocab=<file>
    --all=<directory>
    --output=<file>
"""

import logging
from concurrent import futures
from pathlib import Path

import numpy as np
from numpy.compat import is_pathlib_path

from cli_utils import parse_args
from deeprel import utils
from deeprel.model import re_vocabulary


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

        if len(x) > self.vocabs.max_len:
            logging.warning('Sentence has more than {} words: {}'.format(self.vocabs.max_len, len(x)))
            x = x[:self.vocabs.max_len]
            x_text = x_text[:self.vocabs.max_len]

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

    def create_matrix_asyn(self, source, toks_name='toks'):
        assert toks_name in ('toks', 'shortest path')
        total_cnt = 0
        with futures.ProcessPoolExecutor(max_workers=64) as exec:
            fs = {}
            for obj, ex in utils.example_iterator([source], Path(self.json_dir)):
                if 'toks' in ex and 'shortest path' in ex:
                    f = exec.submit(self.transformer.transform, ex[toks_name], ex['label'])
                    fs[f] = (total_cnt, obj, ex)
                    total_cnt += 1

            xs = [None] * total_cnt
            ys = [None] * total_cnt
            xs_text = [None] * total_cnt

            for f in futures.as_completed(fs):
                i, obj, ex = fs[f]
                try:
                    x, y, x_text = f.result()
                    xs[i] = x
                    ys[i] = y
                    xs_text[i] = (ex['id'], x_text)
                except:
                    logging.exception('{}'.format(obj['id']))
                    exit(1)

            # check None
            for i, x in enumerate(xs):
                assert x is not None, 'Error in {}' % i
            for i, x in enumerate(ys):
                assert x is not None, 'Error in {}' % i
            for i, x in enumerate(xs_text):
                assert x is not None, 'Error in {}' % i

            self.x = np.stack(xs, axis=0)

            labels = np.stack(ys, axis=0)
            self.y = np.zeros((labels.size, len(self.vocabs.label_vocab)))
            self.y[np.arange(labels.size), labels] = 1

            self.xs_text = xs_text

            logging.debug('x shape: %s', self.x.shape)
            logging.debug('y shape: %s', self.y.shape)

    def create_matrix(self, source, toks_name='toks'):
        assert toks_name in ('toks', 'shortest path')

        xs = []
        ys = []
        xs_text = []

        for obj, ex in utils.example_iterator([source], Path(self.json_dir)):
            if 'toks' in ex and 'shortest path' in ex:
                try:
                    x, y, x_text = self.transformer.transform(ex[toks_name], ex['label'])
                    xs.append(x)
                    ys.append(y)
                    xs_text.append((ex['id'], x_text))
                except:
                    logging.exception('{}: {}, {}'.format(obj['id'], ex[toks_name], ex['label']))
                    exit(1)

        self.x = np.stack(xs, axis=0)

        labels = np.stack(ys, axis=0)
        self.y = np.zeros((labels.size, len(self.vocabs.label_vocab)))
        self.y[np.arange(labels.size), labels] = 1

        self.xs_text = xs_text

        logging.debug('x shape: %s', self.x.shape)
        logging.debug('y shape: %s', self.y.shape)

    def save(self, dst, verbose=False):
        np.savez(dst, x=self.x, y=self.y)

        if verbose:
            if is_pathlib_path(dst):
                dst = str(dst)
            with open(dst + '.txt', 'w') as f:
                for docid, x_text in self.xs_text:
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
    if argv['matrix']:
        mc.create_matrix_asyn(argv['INPUT_FILE'], toks_name='toks')
    elif argv['sp']:
        mc.create_matrix_asyn(argv['INPUT_FILE'], toks_name='shortest path')
    mc.save(argv['--output'])
