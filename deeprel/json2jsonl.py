"""
Usage:
    json2jsonl [options] <input>...

Options:
    --verbose
"""
import json
import logging

import docopt
import ijson
from tqdm import tqdm

from utils import to_path


def json_iterator(source, verbose=True):
    logging.debug('Processing %s', source)
    with open(source) as fp:
        item_itr = ijson.items(fp, 'item')
        if verbose:
            item_itr = tqdm(item_itr, unit='item')
        for obj in item_itr:
            yield obj


if __name__ == '__main__':
    argv = docopt.docopt(__doc__)

    for input in argv['<input>']:
        input = to_path(input)
        dest = input.with_suffix('.jsonl')
        with open(input) as fin, open(dest, 'w') as fout:
            item_itr = ijson.items(fin, 'item')
            if argv['--verbose']:
                item_itr = tqdm(item_itr, unit='item')
            for obj in item_itr:
                fout.write(json.dumps(obj) + '\n')
