"""
Usage:
    create_sp_matrix.py [options] --vocab=<file> --all=<directory> --output=<file> INPUT_FILE

Options:
    --verbose
    --vocab=<file>
    --all=<directory>
    --output=<file>
"""

import collections
import json
import logging
import os
import sys
from pathlib import Path

import docopt

from cli_utils import parse_args
from deeprel import utils
from deeprel.create_matrix import MatrixCreater


class SpMatrixCreater(MatrixCreater):
    def __init__(self, vocabs_filename, json_dir):
        super(SpMatrixCreater, self).__init__(vocabs_filename, json_dir)

    def create_matrix(self, src, dst):
        xs = []
        ys = []
        xs_text = []

        for obj, ex in utils.example_iterator([src], Path(self.json_dir)):
            if 'toks' in ex and 'shortest path' in ex:
                x, y, x_text = self.transformer.transform(ex['shortest path'], ex['label'])
                xs.append(x)
                ys.append(y)
                xs_text.append((ex['id'], x_text))
        self.save(xs, ys, xs_text, dst)


if __name__ == '__main__':
    argv = parse_args(__doc__)
    mc = SpMatrixCreater(argv['--vocab'], argv['--all'])
    mc.create_matrix(argv['INPUT_FILE'], argv['--output'])