"""
Usage:
    universal_sentence [options] --output=<file> --input=<file>

Options:
    --verbose
    --output=<file>
    --input=<file>
    --skip
"""

import logging
from os import PathLike
from pathlib import Path
from typing import List

import numpy as np

from deeprel import utils
from utils import parse_args, to_path

MODULE_URL = "/panfs/pan1.be-md.ncbi.nlm.nih.gov/bionlp/lulab/pengy6/data/embedding/models/sent2vec/pubmed_NOTEEVENTS-bigram.bin"


def read_corpus(filename: Path) -> List[str]:
    messages = []
    for i, (obj, ex) in enumerate(utils.example_iterator2(filename)):
        words = []
        for tok in ex['toks']:
            if tok['type'] == 'O':
                words.append(tok['word'])
            else:
                words.append(tok['type'])
        messages.append(' '.join(words))
    return messages


def transform(src: Path, dst: PathLike):
    import sent2vec
    logging.info('Reading %s', MODULE_URL)
    model = sent2vec.Sent2vecModel()
    model.load_model(MODULE_URL)

    test_corpus = read_corpus(src)
    x = []
    for sent in test_corpus:
        try:
            # (1, D)
            t = model.embed_sentence(sent)
        except:
            t = np.zeros((1, 700))
        x.append(t)
    x = np.concatenate(x, axis=0)
    logging.info('Shape %s', x.shape)
    logging.info('Save to %s', dst)
    np.savez(dst, x=x)


def read_s2v_sentence(filename):
    npzfile = np.load(filename)
    return npzfile['x']


if __name__ == '__main__':
    argv = parse_args(__doc__)

    # pick_device()

    output = to_path(argv['--output'])
    if not argv['--skip'] or not output.exists():
        transform(to_path(argv['--input']), output)
