"""
Usage: 
    create_features.py [options] --output=<directory> INPUT_FILE...

Options:
    --verbose
    --asyn                  asynchronous execution
    --output=<directory>
"""

import json
import logging
import sys
from pathlib import Path

import docopt
import os
import collections
from concurrent import futures

from cli_utils import parse_args
from deeprel.preprocessor import feature
from deeprel import utils


def one_file(source):
    logging.debug('Process: %s', source)
    with open(source) as fp:
        obj = json.load(fp, object_pairs_hook=collections.OrderedDict)
    obj['examples'] = feature.generate(obj)
    with open(source, 'w') as fp:
        json.dump(obj, fp, indent=2)


def syn_process(arguments):
    output_dir = Path(arguments['--output'])
    for obj in utils.json_iterator(arguments['INPUT_FILE']):
        docid = obj['id']
        jsonfile = output_dir / (docid + '.json')
        if not jsonfile.exists():
            logging.warning('Cannot find file %s', jsonfile)
        else:
            try:
                one_file(jsonfile)
            except:
                logging.exception('%s generated an exception', jsonfile)


def asyn_process(arguments):
    output_dir = Path(arguments['--output'])
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_map = {}

        for obj in utils.json_iterator(arguments['INPUT_FILE']):
            docid = obj['id']
            jsonfile = output_dir / (docid + '.json')
            if not jsonfile.exists():
                logging.warning('Cannot find file %s', jsonfile)
            else:
                future_map[executor.submit(one_file, jsonfile)] = jsonfile

        for future in futures.as_completed(future_map):
            srcfile = future_map[future]
            try:
                future.result()
            except:
                logging.exception('%s generated an exception', srcfile)


if __name__ == '__main__':
    argv = parse_args(__doc__)
    if argv['--asyn']:
        asyn_process(argv)
    else:
        syn_process(argv)
