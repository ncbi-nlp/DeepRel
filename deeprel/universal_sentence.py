"""
Usage:
    universal_sentence transform [options] --all=<directory> --output=<file> INPUT_FILE

Options:
    --verbose
    --output=<file>
    --all=<directory>
    --skip
"""

import logging
from os import PathLike
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from cli_utils import parse_args
from deeprel import utils
from utils import pick_device

MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/2"


def read_corpus(jsondir: Path, filename: Path) -> List[str]:
    messages = []
    for i, (obj, ex) in enumerate(utils.example_iterator([filename], jsondir)):
        words = []
        for tok in ex['toks']:
            if tok['type'] == 'O':
                words.append(tok['word'])
            else:
                words.append(tok['type'])
        messages.append(' '.join(words))
    return messages


def transform(jsondir: Path, src: Path, dst: PathLike):
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(MODULE_URL)
    tf.logging.set_verbosity(tf.logging.ERROR)
    logging.debug('Transforming')
    test_corpus = read_corpus(jsondir, src)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        x = session.run(embed(test_corpus))
    logging.debug('Save to %s', dst)
    np.savez(dst, x=x)


def read_universal_sentence(filename):
    npzfile = np.load(filename)
    return npzfile['x']


if __name__ == '__main__':
    argv = parse_args(__doc__)

    all = Path(argv['--all'])
    assert all.exists(), '%s does not exist' % all

    # pick_device()

    if argv['transform']:
        output = Path(argv['--output'])
        if not argv['--skip'] or not output.exists():
            transform(all, Path(argv['INPUT_FILE']), output)
