import json
import logging
import os
import random
import string
import tempfile
from concurrent import futures
from pathlib import Path
from subprocess import call
from typing import Tuple, List

import docopt
import ijson
import numpy as np
import pandas as pd
import psutil
import tqdm
from numpy.compat import is_pathlib_path

from deeprel import metrics


def example_iterator2(*files, verbose=True):
    with tqdm.tqdm(unit=' examples', disable=not verbose) as pbar:
        for input_file in files:
            logging.debug('Processing %s', input_file)
            with open(input_file) as fp:
                for line in fp:
                    obj = json.loads(line)
                    try:
                        for ex in obj['examples']:
                            yield obj, ex
                    except:
                        logging.exception('Cannot find examples: %s', obj['id'])
                        exit(1)
                    pbar.update(len(obj['examples']))


def to_path(s):
    if isinstance(s, np.compat.basestring):
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


def random_text(k=3):
    """
    Returns random text of length k
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))


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


def submit_cmds(cmds, max_workers=4):
    with futures.ProcessPoolExecutor(max_workers=max_workers) as exec:
        future_map = {}
        for cmd in cmds:
            logging.info('Submit %s', cmd)
            f = exec.submit(call, cmd, shell=True)
            future_map[f] = cmd
        for f in futures.as_completed(future_map):
            cmd = future_map[f]
            try:
                r = f.result()
            except:
                logging.error('Failed: %s', cmd)
            else:
                logging.debug('Return: %s, %s', r, cmd)


def precess_jsonl(source, dest, func, verbose=True):
    dest = to_path(dest)
    error = dest.parent / '{}-errorlines.jsonl'.format(dest.stem)
    with open(source) as fin, open(dest, 'w') as fout, open(error, 'w') as ferr:
        if verbose:
            fin = tqdm.tqdm(fin, unit='lines')
        for i, line in enumerate(fin):
            line = line.strip()
            try:
                obj = json.loads(line)
                obj = func(obj)
                line = json.dumps(obj)
            except:
                logging.error('Line %s returns errors', i)
                ferr.write(line + '\n')
            else:
                fout.write(line + '\n')


def parse_args(doc, **kwargs):
    argv = docopt.docopt(doc, **kwargs)

    argv['--verbose'] = int(argv['--verbose'])
    if argv['--verbose'] == 2:
        logging.basicConfig(level=logging.DEBUG)
    elif argv['--verbose'] == 1:
        logging.basicConfig(level=logging.INFO)
    elif argv['--verbose'] == 0:
        logging.basicConfig(level=logging.WARNING)
    else:
        raise KeyError

    s = ''
    for k, v in argv.items():
        s += '    {}: {}\n'.format(k, v)
    logging.debug('Arguments:\n%s', s)
    return argv

