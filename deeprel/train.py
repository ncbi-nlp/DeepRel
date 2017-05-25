"""
Usage: 
    train.py [options] INI_FILE
    
Options:
    --log <str>     Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
"""
from __future__ import print_function

import configparser
import json
import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime

import docopt
import numpy as np
import tensorflow as tf
from sklearn import metrics

from deeprel import create_matrix
from deeprel.create_vocabs import VocabsCreater
from deeprel.model import re_vocabulary
from deeprel.model.cnn_model import CnnModel


def read_vocabs(config):
    filename = os.path.join(config['model_dir'], VocabsCreater.vocab_file)
    logging.info('Read vocabs: %s', filename)
    vocabs = re_vocabulary.load(filename)
    return vocabs


def read_embeddings(config, embeddings):
    matrices = []
    for embedding in embeddings:
        filename = os.path.join(config['model_dir'], embedding)
        logging.info('Loading embeddings: %s', filename)
        with open(filename) as f:
            npzfile = np.load(f)
            matrix = npzfile['embeddings']
            matrices.append(matrix)
            logging.info('%s shape: %s', embedding, matrix.shape)
    return matrices


def read_doc2vec(filename):
    with open(filename) as f:
        npzfile = np.load(f)
        return npzfile['x']


def prepare_config(config):
    """
    Returns:
        dict: new config
    """
    newconfig = OrderedDict({
        'model_dir': config['deeprel']['model_dir'],
        'cnn_config': os.path.join(config['deeprel']['model_dir'], 'cnn_model_config.json'),
        'log_dir': config['deeprel']['model_dir'],
        'checkpoint_dir': config['deeprel']['model_dir'],
        #
        'train_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['training_set'] + '.npz'),
        'train_sp_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['training_set'] + '-sp.npz'),
        'train_doc_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['training_set'] + '-doc.npz'),
        #
        'dev_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['dev_set'] + '.npz'),
        'dev_sp_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['dev_set'] + '-sp.npz'),
        'dev_doc_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['dev_set'] + '-doc.npz'),
        #
        'test_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['test_set'] + '.npz'),
        'test_sp_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['test_set'] + '-sp.npz'),
        'test_doc_matrix': os.path.join(config['deeprel']['model_dir'], config['deeprel']['test_set'] + '-doc.npz'),
    })
    return newconfig


def main(argv):
    # print(argv)
    arguments = docopt.docopt(__doc__, argv=argv)
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    logging.info('Read config: %s', arguments['INI_FILE'])
    oldconfig = configparser.ConfigParser()
    oldconfig.read(arguments['INI_FILE'])

    embeddings = [VocabsCreater.word_embedding_file,
                  VocabsCreater.pos_embedding_file,
                  VocabsCreater.chunk_embedding_file,
                  VocabsCreater.arg1_dis_embedding_file,
                  VocabsCreater.arg2_dis_embedding_file,
                  VocabsCreater.type_embedding_file,
                  VocabsCreater.dependency_embedding_file]

    newconfig = prepare_config(oldconfig)
    # read vocabs
    vocabs = read_vocabs(newconfig)
    # read embeddings
    matrices = read_embeddings(newconfig, embeddings)

    x_train, y_train = create_matrix.read_matrix(newconfig['train_matrix'])
    x_dev, y_dev = create_matrix.read_matrix(newconfig['dev_matrix'])
    logging.debug('len y_train: %s', len(y_train))
    logging.debug('len y_dev:   %s', len(y_dev))

    x_sp_train, y_sp_train = create_matrix.read_matrix(newconfig['train_sp_matrix'])
    x_sp_dev, y_sp_dev = create_matrix.read_matrix(newconfig['dev_sp_matrix'])
    logging.debug('len y_sp_train: %s', len(y_sp_train))
    logging.debug('len y_sp_dev:   %s', len(y_sp_dev))

    x_global_train = read_doc2vec(newconfig['train_doc_matrix'])
    x_global_dev = read_doc2vec(newconfig['dev_doc_matrix'])
    logging.debug('x_global_train shape: {}'.format(x_global_train.shape))
    logging.debug('x_global_dev shape: {}'.format(x_global_dev.shape))

    newconfig.update(
        {
            'n_features': 7,
            'seq_len': vocabs.max_len,
            'n_classes': len(vocabs.label_vocab),
            'lr': 7e-4,
            'window_size': 3,

            'num_filters': 400,
            'l2_reg_lambda': 0,
            'num_epochs': 250,
            'batch_size': 128,
            'training_keep_prob': 0.5,
            'validate_keep_prob': 1,

            'w_emb_size': matrices[0].shape[1],
            'pos_emb_size': matrices[1].shape[1],
            'chunk_emb_size': matrices[2].shape[1],
            'dis1_emb_size': matrices[3].shape[1],
            'dis2_emb_size': matrices[4].shape[1],
            'type_emb_size': matrices[5].shape[1],
            'dependency_emb_size': matrices[6].shape[1],
            'doc_emb_size': 200,
        }
    )

    cnn_config_file = newconfig['cnn_config']
    logging.info('CNN config: \n%s', json.dumps(newconfig, indent=2))
    with open(cnn_config_file, 'w') as f:
        json.dump(newconfig, f, indent=2)

    with tf.Graph().as_default():
        cnn = CnnModel(newconfig,
                       matrices[0],
                       matrices[1],
                       matrices[2],
                       matrices[3],
                       matrices[4],
                       matrices[5],
                       matrices[6])
        init = tf.global_variables_initializer()

        log_dir = newconfig['log_dir']
        checkpoint_dir = newconfig['checkpoint_dir']

        with tf.Session() as session:
            # Training :
            print(80 * "=")
            print("TRAINING")
            print(80 * "=")
            session.run(init)

            # trainable variables
            vars = tf.trainable_variables()
            for v in vars:
                print(v)

            best_dev_f1 = -1

            cnn.summary_writer = tf.summary.FileWriter(log_dir, session.graph)
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Loading the latest model')
                saver.restore(session, ckpt.model_checkpoint_path)
                val_loss, y_pred = cnn.predict(session, x_dev, x_sp_dev, x_global_dev, y_dev)
                best_dev_f1 = cid_f1_score(np.argmax(y_dev, 1),
                                           np.argmax(y_pred, 1))
                logging.info('Best f1: {:.5f}'.format(best_dev_f1))

            prev_epoch_loss = float('inf')
            for epoch in range(newconfig['num_epochs']):
                train_loss, train_true, train_pred = cnn.run_epoch(
                    session, x_train, x_sp_train, x_global_train, y_train, shuffle=False,
                    epoch=epoch)
                # print(train_loss, train_acc)
                val_loss, y_pred = cnn.predict(session, x_dev, x_sp_dev, x_global_dev, y_dev)
                # print('y_dev ', np.argmax(y_dev, 1))
                # print('y_pred', np.argmax(y_pred, 1))

                # lr annealing
                # if val_loss > prev_epoch_loss * config['anneal_threshold']:
                #     config['lr'] /= config['anneal_by']
                #     print('annealed lr to %f' % config['lr'])
                # prev_epoch_loss = val_loss

                now = datetime.now()
                train_f1 = cid_f1_score(np.argmax(train_true, 1),
                                        np.argmax(train_pred, 1))
                print("{}: Epoch {:02d}, train loss {:.5f}, train f1 {:.5f}".format(
                    now.strftime("%Y-%m-%d %H:%M:%S"), epoch, train_loss, train_f1))

                val_f1 = cid_f1_score(np.argmax(y_dev, 1),
                                      np.argmax(y_pred, 1))
                print("{}: Epoch {:02d}, val loss   {:.5f}, val_f1   {:.5f}".format(
                    now.strftime("%Y-%m-%d %H:%M:%S"), epoch, val_loss, val_f1))

                if best_dev_f1 < val_f1:
                    path = saver.save(
                        session,
                        os.path.join(checkpoint_dir, 'cnn_mdoel'),
                        global_step=cnn.global_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    best_dev_f1 = val_f1
    logging.info('Done. The model is saved at %s', newconfig['model_dir'])


def cid_f1_score(y_true, y_pred):
    try:
        return metrics.f1_score(y_true, y_pred)
    except:
        logging.exception('Cannot calculate f1')
        return 0


if __name__ == '__main__':
    main(sys.argv[1:])
