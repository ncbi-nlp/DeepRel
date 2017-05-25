"""
Usage:
    test_matrix.py [options] VOCAB_FILE TRAINING_MATRIX DEV_MATRIX TEST_MATRIX
    
Options:
    --log <str>     Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
"""

from __future__ import division, absolute_import, print_function

import json
import logging
import sys

import docopt

from deeprel import create_matrix
from deeprel.model import re_vocabulary


def test(vocab_file, training_matrix, test_matrix):
    vocabulary = re_vocabulary.load(vocab_file)
    vocabulary.summary()

    x_train, y_train = create_matrix.read_matrix(training_matrix)
    print('train: x shape: {}'.format(x_train.shape))
    assert x_train.shape[0] == y_train.shape[0], \
        'train: x shape: {}, y shape: {}'.format(x_train.shape, y_train.shape)

    x_test, y_test = create_matrix.read_matrix(test_matrix)

    # instances
    assert x_test.shape[0] == y_test.shape[0], \
        'x shape: {}, y shape: {}'.format(x_test.shape, y_test.shape)

    # number of words in one document
    assert vocabulary.max_len == x_train.shape[1] == x_test.shape[1], \
        'x shape: {}, train: x shape: {}, max_len: {}'.format(
            x_test.shape, x_train.shape, vocabulary.max_document_length)

    # label
    assert len(vocabulary.label_vocab) == y_train.shape[1] == y_test.shape[1], \
        'y shape: {}, train: y shape: {}'.format(y_test.shape, y_train.shape)

    # features of each word
    assert len(vocabulary.keys) == x_train.shape[2] == x_test.shape[2], \
        'x shape: {}, train x shape: {}'.format(x_test.shape, x_train.shape)
    # check matrix
    for i, key in enumerate(vocabulary.keys):
        logging.info('Check %s', key)
        for idx in x_train[:, :, i].flatten():
            assert idx < len(vocabulary.vocabs[key]), \
                '{} is larger then len of vocab {}\n{}\n{}'.format(
                    idx, key,
                    json.dumps(vocabulary.vocabs[key].category2index()),
                    json.dumps(vocabulary.vocabs[key].index2category()))

    print('Passed')


def main(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    test(arguments['VOCAB_FILE'], arguments['TRAINING_MATRIX'], arguments['TEST_MATRIX'])
    test(arguments['VOCAB_FILE'], arguments['TRAINING_MATRIX'], arguments['DEV_MATRIX'])


if __name__ == '__main__':
    main(sys.argv[1:])
