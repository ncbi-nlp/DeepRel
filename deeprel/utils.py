from __future__ import print_function

import collections
import json
import logging
import tempfile

import numpy as np

from deeprel import metrics


def json_iterator(files):
    for input_file in files:
        with open(input_file) as fp:
            objs = json.load(fp, object_pairs_hook=collections.OrderedDict)

        idx = 1
        for idx, obj in enumerate(objs, 1):
            logging.debug('Process: %s', obj['id'])
            if idx % 500 == 0:
                logging.info('Process: %s documents', idx)
            yield obj
        logging.info('Process: %s documents', idx)


def intersect(range1, range2):
    """
    Args:
        range1(int, int): [begin, end)
        range2(int, int): [begin, end)
    """
    if range1[0] <= range2[0] < range1[1]:
        return True
    elif range1[0] < range2[1] <= range1[1]:
        return True
    elif range2[0] <= range1[0] < range2[1]:
        return True
    elif range2[0] < range1[1] <= range2[1]:
        return True
    return False


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    Args:
        l(list): a list
        n(int): size of chunks

    Return:
        a chunk
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_tempfile(suffix):
    """
    Create a temporary file.

    Args:
        suffix(str):

    Return:
        file name
    """
    fp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fp.close()
    return fp.name


def data_iterator(orig_x, origin_x_sp, orign_x_global, orig_y=None, batch_size=32, shuffle=False):
    # Optionally shuffle the data before training
    data_size = len(orig_x)
    if shuffle:
        indices = np.random.permutation(data_size)
        x = orig_x[indices]
        x_sp = origin_x_sp[indices]
        x_global = orign_x_global[indices]
        y = orig_y[indices] if np.any(orig_y) else None
    else:
        x = orig_x
        x_sp = origin_x_sp
        x_global = orign_x_global
        y = orig_y
    ###

    total_processed_examples = 0
    total_steps = int((data_size - 1) / batch_size) + 1
    for step in range(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        batch_end = min(batch_start + batch_size, data_size)
        subx = x[range(batch_start, batch_end)]
        subx_sp = x_sp[range(batch_start, batch_end)]
        subx_global = x_global[range(batch_start, batch_end)]
        suby = None
        if np.any(y):
            suby = y[batch_start:batch_start + batch_size]
        ###
        yield subx, subx_sp, subx_global, suby
        total_processed_examples += len(subx)
    # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == len(x), \
        'Expected {} and processed {}'.format(len(x),
                                              total_processed_examples)


def print_confusion(confusion, vocab):
    """
    Helper method that prints confusion matrix.
    """
    # Summing top to bottom gets the total number of tags guessed as T
    total_guessed_tags = confusion.sum(axis=0)
    # Summing left to right gets the total number of true tags
    total_true_tags = confusion.sum(axis=1)
    print('Eval report:')
    total = {}
    for i in range(len(vocab)):
        total[vocab.reverse(i)] = [
            confusion[i, i],
            total_guessed_tags[i] - confusion[i, i],
            total_true_tags[i] - confusion[i, i]]
    print(metrics.classification_report(total, type='markdown'))
