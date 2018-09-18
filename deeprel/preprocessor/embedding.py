import logging
import time

import numpy as np
from gensim.models import KeyedVectors
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


def get_word_embeddings(src, vocab, dst):
    """
    Create np word2vec matrix based on vocabs
    """
    start = time.time()
    word_vectors = KeyedVectors.load_word2vec_format(src, binary=True)
    logging.info("took {:.2f} seconds".format(time.time() - start))
    logging.info("vector size: %s", word_vectors.vector_size)
    logging.info("vocab  size: %s", len(word_vectors.vocab))

    matrix = np.zeros((len(vocab), word_vectors.vector_size), dtype=np.float32)
    # matrix = np.asarray(
    #     np.random.normal(0, 0.9, (len(vocab), word_vectors.vector_size)), dtype=np.float32)
    for idx in range(len(vocab)):
        token = vocab.reverse(idx)
        if token in word_vectors:
            matrix[idx] = word_vectors[token]
        elif token.lower() in word_vectors:
            matrix[idx] = word_vectors[token.lower()]
        else:
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
