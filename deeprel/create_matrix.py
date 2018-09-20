"""
Usage:
    create_matrix.py matrix [options] --vocab=<file> --input=<file> --output=<file>
    create_matrix.py sp [options] --vocab=<file> --input=<file> --output=<file>
    
Options:
    --verbose
    --vocab=<file>
    --output=<file>
    --input=<file>
"""
import functools
import logging
from concurrent import futures

import numpy as np
from numpy.compat import is_pathlib_path

from deeprel import utils
from deeprel.model import re_vocabulary
from model.re_vocabulary import ReVocabulary
from utils import parse_args


def transform(ex, toks_name, vocabs: ReVocabulary):
    """Transform toks to indices.

    Return:
        x: ('number of words in a sentence', 6)
        y: ('number of words in a sentence', )
        list: token['word']
    """
    sentence_max_len = vocabs.max_len
    token_types = len(vocabs.keys)

    x = np.zeros(shape=(sentence_max_len, token_types), dtype=np.int64)
    y = vocabs.label_vocab.get(ex['label'])
    x_text = []  # for debug
    for i, tok in enumerate(ex[toks_name]):
        if i > sentence_max_len:
            logging.warning('Sentence has more than {} words: {}'.format(sentence_max_len, len(ex[toks_name])))
            break

        for j, key in enumerate(vocabs.keys):
            if 'ROOT' in tok and key == 'dependency':
                continue
            v = vocabs[key]
            try:
                idx = v.get(str(tok[key]))
            except:
                logging.debug('{} does not have {}'.format(tok, key))
            else:
                x[i, j] = idx

        x_text.append((i, [tok['word']]))
    return x, y, x_text


# def create_matrix(source, transform_func):
#     xs = []
#     ys = []
#     xs_text = []
#
#     for obj, ex in utils.example_iterator2(source):
#         try:
#             x, y, x_text = transform_func(ex)
#         except:
#             logging.exception('{} {} {}'.format(obj['id'], ex['id'], ex['label']))
#             exit(1)
#         else:
#             xs.append(x)
#             ys.append(y)
#             xs_text.append((ex['id'], x_text))
#
#     x = np.stack(xs, axis=0)
#     y = np.stack(ys, axis=0)
#     xs_text = xs_text
#
#     logging.debug('x shape: %s', x.shape)
#     logging.debug('y shape: %s', y.shape)
#     logging.debug('text shape: %s', len(xs_text))
#
#     return x, y, xs_text


def save(dest, x, y, x_text=None):
    np.savez(dest, x=x, y=y)

    if x_text is not None:
        if is_pathlib_path(dest):
            dest = str(dest)
        with open(dest + '.txt', 'w') as f:
            for docid, x_text in x_text:
                s = docid + '\n'
                for idx, k in x_text:
                    s += '{} = {}\n'.format(idx, ' <-- '.join(k))
                f.write(s)


def create_matrix(source, transform_func, max_workers=1):
    total_cnt = 0
    with futures.ProcessPoolExecutor(max_workers=max_workers) as exec:
        fs = {}
        for obj, ex in utils.example_iterator2(source):
            f = exec.submit(transform_func, ex)
            fs[f] = (total_cnt, obj, ex)
            total_cnt += 1

        xs = [None] * total_cnt
        ys = [None] * total_cnt
        xs_text = [None] * total_cnt

        for f in futures.as_completed(fs):
            i, obj, ex = fs[f]
            try:
                x, y, x_text = f.result()
            except:
                logging.exception('{} {} {}'.format(obj['id'], ex['id'], ex['label']))
                exit(1)
            else:
                xs[i] = x
                ys[i] = y
                xs_text[i] = (ex['id'], x_text)

    # check None
    for i, x in enumerate(xs):
        assert x is not None, 'Error in {}' % i
    for i, x in enumerate(ys):
        assert x is not None, 'Error in {}' % i
    for i, x in enumerate(xs_text):
        assert x is not None, 'Error in {}' % i

    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    xs_text = xs_text

    logging.debug('x shape: %s', x.shape)
    logging.debug('y shape: %s', y.shape)
    logging.debug('text shape: %s', len(xs_text))

    return x, y, xs_text


def read_matrix(src):
    logging.info('Read matrix: %s', src)
    npzfile = np.load(src)
    return npzfile['x'], npzfile['y']


if __name__ == '__main__':
    argv = parse_args(__doc__)
    vocabs = re_vocabulary.load(argv['--vocab'])
    if argv['matrix']:
        func = functools.partial(transform, toks_name='toks', vocabs=vocabs)
    elif argv['sp']:
        func = functools.partial(transform, toks_name='shortest path', vocabs=vocabs)
    else:
        raise KeyError

    x, y, x_text = create_matrix(argv['--input'], func)

    y_cat = np.zeros((y.size, len(vocabs.label_vocab)))
    y_cat[np.arange(y.size), y] = 1

    if argv['--verbose']:
        save(argv['--output'], x, y_cat, x_text)
    else:
        save(argv['--output'], x, y_cat)
