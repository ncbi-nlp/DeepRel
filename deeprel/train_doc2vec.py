"""
Usage:
    train_doc2vec [options] JSON_DIR TRAINING_FILE MODEL

Options:
    --log <str>     Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
    -s <size>       Embedding size. [default: 200]
"""

import json
import logging
import docopt
import sys

import gensim
import os
from deeprel import utils


def read_corpus(jsondir, filename):
    for obj in utils.json_iterator([filename]):
        docid = obj['id']
        filename = os.path.join(jsondir, '{}.json'.format(docid))
        with open(filename) as fp:
            obj = json.load(fp)
        for i, ex in enumerate(obj['examples']):
            words = []
            for tok in ex['toks']:
                if tok['type'] == 'O':
                    words.append(tok['word'])
                else:
                    words.append(tok['type'])
            yield gensim.models.doc2vec.TaggedDocument(words, ['{}-{}'.format(docid, i)])


def train(model_file, jsondir, train_file, embedding_size):
    train_corpus = list(read_corpus(jsondir, train_file))
    logging.debug('Read %s documents for training', len(train_corpus))
    model = gensim.models.doc2vec.Doc2Vec(size=embedding_size, min_count=1, iter=55)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(model_file)


def main(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print('create_matrix')
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    train(arguments['MODEL'], arguments['JSON_DIR'], arguments['TRAINING_FILE'],
          int(arguments['-s']))


if __name__ == '__main__':
    main(sys.argv[1:])
