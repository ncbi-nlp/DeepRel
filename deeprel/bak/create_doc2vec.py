"""
Usage:
    create_doc2vec [options] --all=<directory> --model=<file> --output=<file> INPUT_FILE

Options:
    --verbose
    --model=<file>
    --output=<file>
    --all=<directory>
"""

import logging

import gensim
import numpy as np

from cli_utils import parse_args
from deeprel import train_doc2vec


def create_matrix(model_file, jsondir, src, dst):
    model = gensim.models.doc2vec.Doc2Vec.load(model_file)
    test_corpus = list(train_doc2vec.read_corpus(jsondir, src))
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
    with open(dst, 'wb') as f:
        np.savez(f, x=x)


if __name__ == '__main__':
    argv = parse_args(__doc__)
    create_matrix(argv['--model'], argv['--all'], argv['INPUT_FILE'], argv['--output'])
