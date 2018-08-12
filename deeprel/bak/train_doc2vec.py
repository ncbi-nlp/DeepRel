"""
Usage:
    train_doc2vec [options] JSON_DIR TRAINING_FILE MODEL

Options:
    --verbose
    --size <int>       Embedding size. [default: 200]
"""

import json
import logging

import gensim
import numpy as np
import tqdm

from cli_utils import parse_args
from deeprel import utils


def read_corpus(jsondir, filename):
    ex_index = 0
    with tqdm.tqdm(unit='examples') as pbar:
        for obj in utils.json_iterator([filename], verbose=False):
            docid = obj['id']
            filename = jsondir / '{}.json'.format(docid)
            with open(filename) as fp:
                obj = json.load(fp)
            for i, ex in enumerate(obj['examples']):
                words = []
                for tok in ex['toks']:
                    if tok['type'] == 'O':
                        words.append(tok['word'])
                    else:
                        words.append(tok['type'])
                yield gensim.models.doc2vec.TaggedDocument(words, ['{}-{}-{}'.format(docid, i, ex_index)])
                ex_index += 1
            pbar.set_postfix(file=filename.stem, refresh=False)
            pbar.update(len(obj['examples']))


def fit(model_file, jsondir, train_file, embedding_size):
    train_corpus = list(read_corpus(jsondir, train_file))
    logging.debug('Read %s documents for training', len(train_corpus))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=embedding_size, min_count=1, epochs=55)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save(model_file)


if __name__ == '__main__':
    argv = parse_args(__doc__)
    fit(argv['MODEL'], argv['JSON_DIR'], argv['TRAINING_FILE'], int(argv['--size']))
