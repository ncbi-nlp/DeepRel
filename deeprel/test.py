"""
Usage: 
    test.py [options] MODEL_DIR
    
Options:
    --log <str>     Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
"""
from __future__ import print_function

import json
import logging
import os
import sys

import docopt
import numpy as np
import tensorflow as tf
from sklearn import metrics

from deeprel import create_matrix
from deeprel import train
from deeprel import utils
from deeprel.create_vocabs import VocabsCreater
from deeprel.model.cnn_model import CnnModel


def main(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    config_file = os.path.join(arguments['MODEL_DIR'], 'cnn_model_config.json')
    with open(config_file) as fp:
        config = json.load(fp)
    logging.info('CNN config: \n%s', json.dumps(config, indent=2))

    vocabs = train.read_vocabs(config)

    embeddings = [VocabsCreater.word_embedding_file,
                  VocabsCreater.pos_embedding_file,
                  VocabsCreater.chunk_embedding_file,
                  VocabsCreater.arg1_dis_embedding_file,
                  VocabsCreater.arg2_dis_embedding_file,
                  VocabsCreater.type_embedding_file,
                  VocabsCreater.dependency_embedding_file]
    matrices = train.read_embeddings(config, embeddings)

    x_test, y_test = create_matrix.read_matrix(config['test_matrix'])
    x_sp_test, y_sp_test = create_matrix.read_matrix(config['test_sp_matrix'])
    x_global_test = train.read_doc2vec(config['test_doc_matrix'])
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

            basename = os.path.splitext(os.path.basename(config['test_matrix']))[0]
            with open(os.path.join(config['model_dir'], basename + '.rst'), 'w') as fp:
                for t, p in zip(y_test, y_pred):
                    fp.write('{}\t{}\n'.format(t, p))


if __name__ == '__main__':
    main(sys.argv[1:])
