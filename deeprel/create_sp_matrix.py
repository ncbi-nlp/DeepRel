"""
Usage:
    create_sp_matrix.py [options] VOCAB_FILE JSON_DIR INPUT_FILE DST_FILE
    
Options:
    --log <str>     Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
"""

import collections
import json
import logging
import os
import sys

import docopt

from deeprel import utils
from deeprel.create_matrix import MatrixCreater


class SpMatrixCreater(MatrixCreater):
    def __init__(self, vocabs_filename, json_dir):
        super(SpMatrixCreater, self).__init__(vocabs_filename, json_dir)

    def create_matrix(self, src, dst):
        xs = []
        ys = []
        xs_text = []

        for obj in utils.json_iterator([src]):
            docid = obj['id']
            jsonfile = os.path.join(self.json_dir, docid + '.json')
            with open(jsonfile) as jfp:
                obj = json.load(jfp, object_pairs_hook=collections.OrderedDict)
            if obj['examples']:
                for ex in obj['examples']:
                    if 'toks' in ex and 'shortest path' in ex:
                        x, y, x_text = self.transformer.transform(ex['shortest path'], ex['label'])
                        xs.append(x)
                        ys.append(y)
                        xs_text.append((ex['id'], x_text))
        self.save(xs, ys, xs_text, dst)


def main(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print('create_matrix')
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    mc = SpMatrixCreater(arguments['VOCAB_FILE'], arguments['JSON_DIR'])
    mc.create_matrix(arguments['INPUT_FILE'], arguments['DST_FILE'])


if __name__ == '__main__':
    main(sys.argv[1:])