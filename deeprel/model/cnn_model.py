import logging

import numpy as np
import tensorflow as tf

from deeprel import utils


class CnnModel(object):
    def __init__(self, config,
                 pretrained_word_embeddings,
                 pretrained_pos_embeddings,
                 pretrianed_chunk_embeddings,
                 pretrained_pos1_embeddings,
                 pretrained_pos2_embeddings,
                 pretrained_type_embeddings,
                 pretrained_dependency_embeddings):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.pretrained_pos_embeddings = pretrained_pos_embeddings
        self.pretrianed_chunk_embeddings = pretrianed_chunk_embeddings
        self.pretrained_pos1_embeddings = pretrained_pos1_embeddings
        self.pretrained_pos2_embeddings = pretrained_pos2_embeddings
        self.pretrained_type_embeddings = pretrained_type_embeddings
        self.pretrained_dependency_embeddings = pretrained_dependency_embeddings

        # placeholder
        self.inputs_placeholder = None
        self.sp_inputs_placeholder = None
        self.global_inputs_placeholder = None
        self.keep_prob = None
        self.labels_placeholder = None

        self.pred = None
        self.loss = None
        self.train_op = None
        self.merged_summary_op = None
        self.summary_writer = None
        self.debug_op = None
        self.l2_loss = tf.constant(0.0, name='l2_loss')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.build()

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.merged_summary_op = self.add_summary_op()
        self.debug_op = self.add_debug_op()

    def add_debug_op(self):
        return {'x': self.x, 'h_pool_flat': self.h_pool_flat, 'scores': self.scores,
                'h_drop': self.h_drop}

    def add_summary_op(self):
        merged_summary_op = tf.summary.merge_all()
        return merged_summary_op

    def summary_shape(self, *tensors):
        for v in tensors:
            self.logger.debug('shape of %s: %s', v.name, v.get_shape())

    def add_placeholders(self):
        # [word, pos, chunk, dis1, dis2, type, channel]
        self.inputs_placeholder = tf.placeholder(
            tf.int32,
            [None, self.config['seq_len'], self.config['n_features']],
            name="word")
        self.sp_inputs_placeholder = tf.placeholder(
            tf.int32,
            [None, self.config['seq_len'], self.config['n_features']],
            name="word")
        self.global_inputs_placeholder = tf.placeholder(
            tf.float32,
            [None, self.config['doc_emb_size']],
            name="word")
        self.labels_placeholder = tf.placeholder(
            tf.float32,
            (None, self.config['n_classes']),
            name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='drop')

    def create_feed_dict(self, sentence_inputs_batch, sp_inputs_batch,
                         global_inputs_batch, labels_batch=None,
                         keep_prob=0.5):
        """
        inputs_batch: [word, pos, chunk, dis1, dis2, type]
        """
        feed = {
            self.inputs_placeholder: sentence_inputs_batch,
            self.sp_inputs_placeholder: sp_inputs_batch,
            self.global_inputs_placeholder: global_inputs_batch,
            self.keep_prob: keep_prob
        }
        if labels_batch is not None:
            feed[self.labels_placeholder] = labels_batch
        return feed

    def add_embedding(self):
        with tf.device('/cpu:0'):
            with tf.variable_scope("embedding"):
                w_word_emb = tf.constant(
                    self.pretrained_word_embeddings,
                    dtype=tf.float32,
                    name="w_word_emb")
                w_pos_emb = tf.constant(
                    self.pretrained_pos_embeddings,
                    dtype=tf.float32,
                    name="w_pos_emb")
                w_chunk_emb = tf.constant(
                    self.pretrianed_chunk_embeddings,
                    dtype=tf.float32,
                    name='w_chunk_emb')
                w_pos1_emb = tf.constant(
                    self.pretrained_pos1_embeddings,
                    dtype=tf.float32,
                    name='w_pos1_emb')
                w_pos2_emb = tf.constant(
                    self.pretrained_pos2_embeddings,
                    dtype=tf.float32,
                    name='w_pos2_emb')
                w_type_emb = tf.constant(
                    self.pretrained_type_embeddings,
                    dtype=tf.float32,
                    name='w_type_emb')
                w_dependency_emb = tf.constant(
                    self.pretrained_dependency_embeddings,
                    dtype=tf.float32,
                    name='w_dependency_emb')

        # batch_size x sentence_length x embedding_length
        # word embedding

        # print(self.inputs_placeholder.shape)
        def __slice(column_index):
            s = tf.slice(self.inputs_placeholder,
                         [0, 0, column_index],
                         [-1, -1, 1])
            s = tf.reshape(s, [-1, self.config['seq_len']])
            return s

        def __slice_sp(column_index):
            s = tf.slice(self.sp_inputs_placeholder,
                         [0, 0, column_index],
                         [-1, -1, 1])
            s = tf.reshape(s, [-1, self.config['seq_len']])
            return s

        def get_emb(slice):
            word_emb = tf.nn.embedding_lookup(
                w_word_emb, slice(0),
                name='embedding_lookup_word_emb')
            # POS embedding
            pos_emb = tf.nn.embedding_lookup(
                w_pos_emb, slice(1),
                name='embedding_lookup_pos_emb')
            # chunk embedding
            chunk_emb = tf.nn.embedding_lookup(
                w_chunk_emb, slice(2),
                name='embedding_lookup_chunk_emb')
            # distance from first entity embedding
            pos1_emb = tf.nn.embedding_lookup(
                w_pos1_emb, slice(3),
                name='embedding_lookup_pos1_emb')
            # distance from second entity embedding
            pos2_emb = tf.nn.embedding_lookup(
                w_pos2_emb, slice(4),
                name='embedding_lookup_pos2_emb')
            # type embedding
            type_emb = tf.nn.embedding_lookup(
                w_type_emb, slice(5),
                name='embedding_lookup_type_emb')
            # dependency embedding
            dependency_emb = tf.nn.embedding_lookup(
                w_dependency_emb, slice(6),
                name='embedding_lookup_dependency_emb')
            # concat
            window = tf.concat(
                [word_emb, pos_emb, chunk_emb, pos1_emb, pos2_emb, type_emb],
                axis=2,
                name='window')
            self.summary_shape(word_emb, pos_emb, chunk_emb, pos1_emb, pos2_emb,
                               type_emb)
            return window

        window = get_emb(__slice)
        window = tf.expand_dims(window, -1, name='window_expanded')
        self.summary_shape(window)

        window_sp = get_emb(__slice_sp)
        window_sp = tf.expand_dims(window_sp, -1, name='sp_window_expanded')
        self.summary_shape(window_sp)
        return window, window_sp

    def add_prediction_op(self):

        x, x_sp = self.add_embedding()
        self.x = x

        emb_size = self.config['w_emb_size'] \
                   + self.config['pos_emb_size'] \
                   + self.config['chunk_emb_size'] \
                   + self.config['dis1_emb_size'] \
                   + self.config['dis2_emb_size'] \
                   + self.config['type_emb_size']
        # + self.config['dependency_emb_size']

        window_size = self.config['window_size']
        with tf.variable_scope("conv-maxpool-%s" % window_size):
            # filter:
            #   sliding_window_size x embedding_length x in_channels x out_channels
            # strides:
            #   how much to shift your filter at each step
            filter_shape = (window_size, emb_size, 1, self.config['num_filters'])
            w_filter = tf.get_variable(
                "W_filter_{}".format(window_size),
                initializer=tf.truncated_normal(filter_shape, stddev=0.1))
            b_filter = tf.get_variable(
                "b_{}".format(window_size),
                initializer=tf.constant(0.1, shape=[self.config['num_filters']]))

            w_sp_filter = tf.get_variable(
                "W_sp_filter_{}".format(window_size),
                initializer=tf.truncated_normal(filter_shape, stddev=0.1))
            b_sp_filter = tf.get_variable(
                "b_sp_{}".format(window_size),
                initializer=tf.constant(0.1, shape=[self.config['num_filters']]))

        conv = tf.nn.conv2d(
            x,
            filter=w_filter,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv_{}".format(window_size))
        # Apply non-linearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b_filter), name="relu_{}".format(window_size))
        # Max pooling over the outputs
        pool = tf.nn.max_pool(
            h,
            ksize=[1, self.config['seq_len'] - window_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool_{}".format(window_size))

        self.summary_shape(h, pool)
        self.h_pool_flat = tf.reshape(pool, [-1, self.config['num_filters']], name='h_pool_flat')

        conv_sp = tf.nn.conv2d(
            x_sp,
            filter=w_sp_filter,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv_sp_{}".format(window_size))
        # Apply non-linearity
        h_sp = tf.nn.relu(tf.nn.bias_add(conv_sp, b_sp_filter), name="relu_sp_{}".format(window_size))
        # Max pooling over the outputs
        pool_sp = tf.nn.max_pool(
            h_sp,
            ksize=[1, self.config['seq_len'] - window_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool_sp_{}".format(window_size))

        self.summary_shape(h_sp, pool_sp)
        self.h_pool_flat_sp = tf.reshape(pool_sp, [-1, self.config['num_filters']], name='h_pool_flat_sp')

        self.summary_shape(self.h_pool_flat, self.h_pool_flat_sp, self.global_inputs_placeholder)
        tmp = tf.concat([self.h_pool_flat, self.h_pool_flat_sp, self.global_inputs_placeholder], axis=1)

        # Add dropout
        self.h_drop = tf.nn.dropout(tmp, self.keep_prob)

        with tf.variable_scope("output"):
            w_full = tf.get_variable(
                "W_full",
                (2 * self.config['num_filters'] + self.config['doc_emb_size'], self.config['n_classes']),
                initializer=tf.contrib.layers.xavier_initializer())
            b_full = tf.get_variable(
                'b_full',
                initializer=tf.constant(0.1, shape=[self.config['n_classes']]))

        self.l2_loss += tf.nn.l2_loss(w_full)
        self.l2_loss += tf.nn.l2_loss(b_full)
        scores = tf.matmul(self.h_drop, w_full) + b_full

        pred = scores
        self.scores = scores

        self.summary_shape(scores)
        return pred

    def add_loss_op(self, pred):
        softmax_ce = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred,
            labels=self.labels_placeholder)
        loss = tf.reduce_mean(softmax_ce) + self.config['l2_reg_lambda'] * self.l2_loss
        tf.summary.scalar('loss', loss)
        return loss

    def add_training_op(self, loss):
        # optimizer = tf.train.AdagradOptimizer(self.config['lr'])
        optimizer = tf.train.AdamOptimizer(self.config['lr'])
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=self.global_step)
        return train_op

    def predict(self, session, sentence_inputs_batch, sp_inputs_batch, global_inputs_batch,
                labels_batch=None, shuffle=False):
        """
        Make predictions from the provided model.
        """
        # If y is given, the loss is also calculated
        # We deactivate dropout by setting it to 1
        losses = []
        results = []
        if np.any(labels_batch):
            data = utils.data_iterator(sentence_inputs_batch, sp_inputs_batch,
                                       global_inputs_batch,
                                       labels_batch,
                                       batch_size=self.config['batch_size'],
                                       shuffle=shuffle)
        else:
            data = utils.data_iterator(sentence_inputs_batch, sp_inputs_batch,
                                       global_inputs_batch,
                                       batch_size=self.config['batch_size'],
                                       shuffle=shuffle)

        for step, (x, x_sp, x_global, y) in enumerate(data):
            feed = self.create_feed_dict(
                sentence_inputs_batch=x,
                sp_inputs_batch=x_sp,
                global_inputs_batch=x_global,
                labels_batch=y,
                keep_prob=self.config['validate_keep_prob'])
            eval_tensors = [self.pred]
            if np.any(y):
                eval_tensors += [self.loss]

            eval_ret = session.run(eval_tensors, feed)
            eval_ret = dict(zip(eval_tensors, eval_ret))

            if np.any(y):
                losses.append(eval_ret[self.loss])
            results.extend(eval_ret[self.pred])

        if len(losses) == 0:
            return 0, results
        return np.mean(losses), results

    def run_epoch(self, session, sentence_inputs_batch, sp_inputs_batch, global_inputs_batch,
                  labels_batch, shuffle=True, epoch=0):
        losses = []
        total_true = []
        total_pred = []
        total_steps = int(np.ceil(len(sentence_inputs_batch) / float(self.config['batch_size'])))

        self.logger.debug('inputs_batch shape: %s', sentence_inputs_batch.shape)
        self.logger.debug('inputs_batch shape: %s', global_inputs_batch.shape)
        self.logger.debug('labels_batch shape: %s', labels_batch.shape)
        self.logger.debug('batch_size: %s', self.config['batch_size'])
        self.logger.debug('total_steps: %s', total_steps)

        data = utils.data_iterator(sentence_inputs_batch, sp_inputs_batch,
                                   global_inputs_batch,
                                   labels_batch,
                                   batch_size=self.config['batch_size'],
                                   shuffle=shuffle)
        for step, (x, x_sp, x_global, y) in enumerate(data):
            feed = self.create_feed_dict(
                sentence_inputs_batch=x,
                sp_inputs_batch=x_sp,
                global_inputs_batch=x_global,
                labels_batch=y,
                keep_prob=self.config['training_keep_prob'])

            eval_tensors = [self.loss, self.train_op, self.pred]
            if step % 200 == 0:
                eval_tensors += [self.merged_summary_op]
            eval_ret = session.run(eval_tensors, feed)
            eval_ret = dict(zip(eval_tensors, eval_ret))

            if self.merged_summary_op in eval_tensors:
                self.summary_writer.add_summary(
                    eval_ret[self.merged_summary_op],
                    epoch * total_steps + step)

            y_pred = eval_ret[self.pred]
            # print('pred: ', y_pred)
            # print('true: ', y)
            # break

            total_true.extend(y)
            total_pred.extend(y_pred)
            losses.append(eval_ret[self.loss])

            # print ('val acc', val_acc)
            # print ('total_correct_examples', total_correct_examples)
            # print ('total_processed_examples', total_processed_examples)
        return np.mean(losses), total_true, total_pred
