"""
Usage: 
    test.py [options] MODEL_CONFIG
    
Options:
    --verbose
    --output
"""
from __future__ import print_function

import json
import logging
import os
import sys
from pathlib import Path

import docopt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

from sent2vec_sentence import read_s2v_sentence
from utils import parse_args
from deeprel import create_matrix
from deeprel import train
from deeprel import utils
from deeprel.model.cnn_model import CnnModel
from universal_sentence import read_universal_sentence
from utils2 import pick_device


def main():
    arguments = parse_args(__doc__)

    pick_device()

    with open(arguments['MODEL_CONFIG']) as fp:
        config = json.load(fp)
    logging.info('CNN config: \n%s', json.dumps(config, indent=2))

    vocabs = train.read_vocabs(config)
    matrices = train.read_embeddings(config)

    x_test, y_test = create_matrix.read_matrix(config['test_matrix'])
    x_sp_test, y_sp_test = create_matrix.read_matrix(config['test_sp_matrix'])
    # x_global_test = read_doc2vec(config['test_doc_matrix'])
    x_global_test = read_s2v_sentence(config['test_s2v_matrix'])
    logging.debug('x_global_test shape: {}'.format(x_global_test.shape))

    with tf.Graph().as_default():
        cnn = CnnModel(config,
                       matrices[0],
                       matrices[1],
                       matrices[2],
                       matrices[3],
                       matrices[4],
                       matrices[5],
                       matrices[6])
        init = tf.global_variables_initializer()
        checkpoint_dir = config['checkpoint_dir']

        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Loading the latest model:', ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print('checkpoint dir: ', checkpoint_dir)
                print('ckpt: ', ckpt)

            # Testing :
            print(80 * "=")
            print("TESTING")
            print(80 * "=")

            __, y_pred = cnn.predict(session, x_test, x_sp_test, x_global_test)

            print('y_test', y_test.shape)
            print('y_test', np.argmax(y_test, 1).shape)
            print('y_pred', len(y_pred))
            print('y_pred', np.argmax(y_pred, 1).shape)
            # print(model.label_id2tok)

            y_pred = np.argmax(y_pred, 1)
            y_test = np.argmax(y_test, 1)

            # print('y_test', y_test)
            # print('y_pred', y_pred)

            confusion = metrics.confusion_matrix(y_test, y_pred)
            utils.print_confusion(confusion, vocabs.label_vocab)

            if arguments['--output']:
                with open(config['test_set']) as fp:
                    objs = json.load(fp)

                rel_index = 0
                for x in objs:
                    for r in x['relations']:
                        gold = r['label']
                        pred = vocabs.label_vocab.reverse(int(y_pred[rel_index]))
                        test = vocabs.label_vocab.reverse(int(y_test[rel_index]))
                        if gold != test:
                            logging.warning('Cannot match relation %s: %s[gold] vs %s[test] vs %s[pred]',
                                            r['id'], gold, test, pred)
                        r['label-predicted'] = pred
                        rel_index += 1

                with open(Path(config['test_set']).with_suffix('.rst.json'), 'w') as fp:
                    json.dump(objs, fp, indent=2)

            # with open(Path(config['test_set']).with_suffix('.rst'), 'w') as fp:
            #     for t, p in zip(y_test, y_pred):
            #         fp.write('{}\t{}\n'.format(t, p))


if __name__ == '__main__':
    main()
