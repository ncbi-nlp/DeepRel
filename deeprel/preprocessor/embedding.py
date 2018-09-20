import logging
import time
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from numpy.compat import basestring

from deeprel.vocabulary import Vocabulary

POSITION_MATRIX = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # <=-31
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # <=-21
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # <=-11
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # <=-6
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # =-5
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # =-4
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # =-3
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # =-2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # =-1
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # =-1
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # =-2
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # =-3
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # =-4
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # =-5
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # <=-6
    [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # <=-11
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # <=-21
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # <=-31
])


WORD2VEC = 'word2vec'
WORD2VEC_BIN = 'word2vec_bin'
FASTTEXT_BIN = 'fasttext_bin'


class WordEmbeddingModel(object):
    @classmethod
    def load(cls, model_type: str, pathname):
        if isinstance(pathname, basestring):
            pathname = Path(pathname)

        print('Load', pathname)
        if model_type == 'word2vec':
            model = KeyedVectors.load_word2vec_format(str(pathname), binary=True)
            return WordEmbeddingModel(WORD2VEC_BIN, model, pathname)
        elif model_type == 'fasttext':
            if pathname.suffix == '.vec':
                model = KeyedVectors.load_word2vec_format(str(pathname), binary=False)
                return WordEmbeddingModel(WORD2VEC, model, pathname)
            elif pathname.suffix == '' or pathname.suffix == '.vecbin':
                model = KeyedVectors.load_word2vec_format(str(pathname), binary=True)
                return WordEmbeddingModel(WORD2VEC_BIN, model, pathname)
            elif pathname.suffix == '.bin':
                import fastText
                model = fastText.load_model(str(pathname))
                return WordEmbeddingModel(FASTTEXT_BIN, model, pathname)
            else:
                raise KeyError
        else:
            raise KeyError

    def __init__(self, model_type, model, pathname):
        self.model_type = model_type
        self.model = model
        self.pathname = pathname

    def get_word_vector(self, word: str):
        if self.model_type in [WORD2VEC, WORD2VEC_BIN]:
            return self.model[word]
        elif self.model_type == FASTTEXT_BIN:
            return self.model.get_word_vector(word)
        else:
            raise KeyError

    @property
    def vector_size(self):
        if self.model_type in [WORD2VEC, WORD2VEC_BIN]:
            return self.model.vector_size
        elif self.model_type == FASTTEXT_BIN:
            return 200
        else:
            raise KeyError


def get_word_embeddings(model_pathname, vocab, dst):
    """
    Create np word2vec matrix based on vocabs
    """
    start = time.time()
    index = model_pathname.find(':')
    word_vectors = WordEmbeddingModel.load(model_pathname[:index], model_pathname[index+1:])
    # word_vectors = KeyedVectors.load_word2vec_format(model_pathname, binary=True)
    logging.info("took {:.2f} seconds".format(time.time() - start))
    logging.info("vector size: %s", word_vectors.vector_size)
    # logging.info("vocab  size: %s", len(word_vectors.vocab))

    matrix = np.zeros((len(vocab), word_vectors.vector_size), dtype=np.float32)
    # matrix = np.asarray(
    #     np.random.normal(0, 0.9, (len(vocab), word_vectors.vector_size)), dtype=np.float32)
    for idx in range(len(vocab)):
        token = vocab.reverse(idx)
        try:
            matrix[idx] = word_vectors.get_word_vector(token)
        except:
            try:
                matrix[idx] = word_vectors.get_word_vector(token.lower())
            except:
                logging.warning('Cannot find %s in word2vec', token)

    logging.info('word embedding matrix shape: %s', matrix.shape)
    np.savez(dst, embeddings=matrix)


def get_pos_embeddings(vocab, dst):
    # One hot
    a = np.zeros((len(vocab),), dtype=np.int8)
    for idx in range(len(vocab)):
        pos = vocab.reverse(idx).lower()
        if pos.startswith('nn'):
            a[idx] = 0
        elif pos.startswith('vb'):
            a[idx] = 1
        elif pos in ['to', 'in', 'cc']:
            a[idx] = 2
        elif pos.startswith('rb') or pos.startswith('jj') or pos in ['dt', 'pdt', 'md', 'mdn']:
            a[idx] = 3
        elif pos.startswith('wp') or pos in ['wrb', 'wdt', 'ex']:
            a[idx] = 4
        elif pos.startswith('prp'):
            a[idx] = 5
        elif pos in ['pos', 'rp', 'fw', 'hyph', 'sym', '.', ',', '(', ')', 'afx', 'ls', '``', "''",
                     'uh', ':']:
            a[idx] = 6
        elif pos.startswith('cd'):
            a[idx] = 7
        else:
            logging.warning('Cannot match POS %s', pos)
            a[idx] = 8
    matrix = np.zeros((a.size, 9), dtype=np.float)
    matrix[np.arange(a.size), a] = 1

    logging.info('pos matrix shape: %s', matrix.shape)
    np.savez(dst, embeddings=matrix)


def get_distance_embeddings(vocab, dst, name=None):
    matrix = np.zeros((len(vocab), POSITION_MATRIX.shape[1]), dtype=np.float32)
    for idx in range(len(vocab)):
        try:
            dis = int(vocab.reverse(idx))
            if dis <= -31:
                dis = 0
            elif dis <= -21:
                dis = 1
            elif dis <= -11:
                dis = 2
            elif dis <= -6:
                dis = 3
            elif dis <= 5:
                dis += 9
            elif dis <= 10:
                dis = 15
            elif dis <= 20:
                dis = 16
            elif dis <= 30:
                dis = 17
            else:
                dis = 18
        except:
            dis = 18
        matrix[idx] = POSITION_MATRIX[dis]

    logging.info('%s matrix shape: %s', name, matrix.shape)
    np.savez(dst, embeddings=matrix)


def get_one_hot(vocab, dst, name=None):
    matrix = np.diag(range(len(vocab)))
    np.savez(dst, embeddings=matrix)
    logging.info('%s matrix shape: %s', name, matrix.shape)


def get_dependency_embeddings(vocab, dst):
    # One hot
    new_vocab = Vocabulary()
    for idx in range(len(vocab)):
        tag = vocab.reverse(idx).lower()
        index = tag.find(':')
        if index != -1:
            tag = tag[:index]
        new_vocab.add(tag)
    new_vocab.freeze()
    matrix = np.zeros((len(vocab), len(new_vocab)), dtype=np.float32)
    for idx in range(len(vocab)):
        tag = vocab.reverse(idx).lower()
        index = tag.find(':')
        if index != -1:
            tag = tag[:index]
        matrix[idx, new_vocab.get(tag)] = 1

    logging.info('dependency matrix shape: %s', matrix.shape)
    np.savez(dst, embeddings=matrix)
