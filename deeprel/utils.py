from __future__ import print_function

import collections
import json
import logging
import os
from pathlib import Path

import psutil
import tempfile
from typing import Tuple, List

import GPUtil
import numpy as np
import pandas as pd
import tqdm
import ijson
from numpy.compat import is_pathlib_path

from deeprel import metrics


def json_iterator(files, verbose=True, **kwargs):
    for input_file in files:
        logging.debug('Processing %s', input_file)
        with open(input_file) as fp:
            for obj in tqdm.tqdm(ijson.items(fp, 'item'), disable=not verbose, **kwargs):
                yield obj


def example_iterator(files: List, jsondir, verbose=True):
    with tqdm.tqdm(unit=' examples', disable=not verbose) as pbar:
        for input_file in files:
            logging.debug('Processing %s', input_file)
            with open(input_file) as fp:
                for obj in ijson.items(fp, 'item'):
                    docid = obj['id']
                    source = jsondir / '{}.json'.format(docid)
                    with open(source) as fp:
                        obj = json.load(fp)
                    try:
                        for ex in obj['examples']:
                            yield obj, ex
                    except:
                        logging.exception('Cannot find examples: %s', obj['id'])
                        exit(1)
                    pbar.set_postfix(file=source.stem)
                    pbar.update(len(obj['examples']))


def to_path(s):
    if isinstance(s, np.basestring):
        return Path(s)
    elif is_pathlib_path(s):
        return s
    raise TypeError('Cannot convert %r to Path' % s)


def intersect(range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
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


def create_tempfile(suffix: str) -> str:
    """
    Create a temporary file.
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
        # print(batch_start, batch_end)
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
        'Expected {} and processed {}'.format(len(x), total_processed_examples)


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
    print(metrics.classification_report(total))

    columns = [vocab.reverse(i) for i in range(len(vocab))]
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(pd.DataFrame(confusion, columns=columns, index=columns))


def pick_device():
    try:
        GPUtil.showUtilization()
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getFirstAvailable()
        DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
        print('Device ID (unmasked): ' + str(DEVICE_ID))
    except:
        logging.exception('Cannot detect GPUs')


def get_max_workers(max_workers=None):
    w = max(len(psutil.Process().cpu_affinity()) - 1, 1)
    if max_workers:
        w = min(w, max_workers)
    return w

    # if hasattr(os, 'sched_getaffinity'):
    #     return min(len(os.sched_getaffinity(0)) - 1, 1)
    # else:
    #     max_cpu = os.cpu_count() or 1
    #     workers = 1
    #     while 2 * workers < max_cpu:
    #         workers *= 2
    #     return workers
