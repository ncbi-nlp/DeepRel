"""
Usage:
    create_doc2vec [options] JSON_DIR MODEL INPUT_FILE OUTPUT_FILE

Options:
    --log <str>     Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
"""

import logging
import sys

import docopt
import gensim
import numpy as np

from deeprel import train_doc2vec


def create_matrix(model_file, jsondir, src, dst):
    model = gensim.models.doc2vec.Doc2Vec.load(model_file)
    test_corpus = {d[1][0]: d for d in train_doc2vec.read_corpus(jsondir, src)}
    logging.debug('Read %s documents for test', len(test_corpus))
    # logging.debug(test_corpus.keys())

    xs = []
    for index, test_doc in enumerate(test_corpus.values(), 1):
        inferred_vector = model.infer_vector(test_doc[0])
        if index == 1:
            logging.debug(test_doc[0])
            logging.debug(inferred_vector)
        xs.append(inferred_vector)
    logging.debug('Len of xs: %s', len(xs))

    x = np.stack(xs, axis=0)
    logging.debug('Save to %s', dst)
    with open(dst, 'w') as f:
        np.savez(f, x=x)


def main(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print('create_matrix')
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    create_matrix(arguments['MODEL'],
                  arguments['JSON_DIR'],
                  arguments['INPUT_FILE'],
                  arguments['OUTPUT_FILE'])


if __name__ == '__main__':
    main(sys.argv[1:])
