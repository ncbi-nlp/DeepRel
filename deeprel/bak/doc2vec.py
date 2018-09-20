"""
Usage:
    doc2vec fit [options] --all=<directory> --model=<file> <input> ...
    doc2vec transform [options] --all=<directory> --model=<file> --output=<file> INPUT_FILE

Options:
    --verbose
    --size <int>       Embedding size. [default: 200]
    --model=<file>
    --output=<file>
    --all=<directory>
    --skip
"""

import json
import logging
from os import PathLike
from pathlib import Path
from typing import List

import gensim
import numpy as np
import tqdm

from utils import parse_args
from deeprel import utils


def read_corpus(filename: Path):
    ex_index = 0
    for i, (obj, ex) in enumerate(utils.example_iterator2(filename)):
        words = []
        for tok in ex['toks']:
            if tok['type'] == 'O':
                words.append(tok['word'])
            else:
                words.append(tok['type'])
        yield gensim.models.doc2vec.TaggedDocument(words, ['{}-{}-{}'.format(obj['id'], i, ex_index)])
        ex_index += 1


def fit(model_file: Path, train_files: List, embedding_size: int):
    corpus = []
    for source in train_files:
        corpus += list(read_corpus(Path(source)))
    logging.debug('Read %s documents for training', len(corpus))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=embedding_size, min_count=1, epochs=55, workers=10)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(str(model_file))


def transform(model_file: Path, src: Path, dst: PathLike):
    model = gensim.models.doc2vec.Doc2Vec.load(str(model_file))
    test_corpus = list(read_corpus(src))
    logging.debug('Read %s documents for test', len(test_corpus))

    xs = []
    for index, test_doc in enumerate(test_corpus, 1):
        inferred_vector = model.infer_vector(test_doc[0])
        if index == 1:
            logging.debug(test_doc[0])
            logging.debug(inferred_vector)
        xs.append(inferred_vector)
    logging.debug('Length of xs: %s', len(xs))

    x = np.stack(xs, axis=0)
    logging.debug('Save to %s', dst)
    np.savez(dst, x=x)


def read_doc2vec(filename):
    npzfile = np.load(filename)
    return npzfile['x']


if __name__ == '__main__':
    argv = parse_args(__doc__)

    model = Path(argv['--model'])

    if argv['fit']:
        if not argv['--skip'] or not model.exists():
            fit(model, argv['<input>'], int(argv['--size']))
    elif argv['transform']:
        assert model.exists(), '%s does not exist' % model
        output = Path(argv['--output'])
        if not argv['--skip'] or not output.exists():
            transform(model, Path(argv['INPUT_FILE']), output)
