"""
Usage: 
    create_features.py [options] OUTPUT_DIR INPUT_FILE...

Options:
    --log <str>     Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
"""

import json
import logging
import sys

import docopt
import os
import collections
from concurrent import futures

from deeprel.preprocessor import feature
from deeprel import utils


def one_file(jsonfile):
    logging.debug('Process: %s', jsonfile)
    with open(jsonfile) as jfp:
        obj = json.load(jfp, object_pairs_hook=collections.OrderedDict)
    obj['examples'] = feature.generate(obj)
    with open(jsonfile, 'w') as jfp:
        json.dump(obj, jfp, indent=2)


def main(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    for obj in utils.json_iterator(arguments['INPUT_FILE']):
        docid = obj['id']
        jsonfile = os.path.join(arguments['OUTPUT_DIR'], docid + '.json')
        if not os.path.exists(jsonfile):
            logging.warning('Cannot find file %s', jsonfile)
        else:
            one_file(jsonfile)


def main_batch(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_map = {}

        for obj in utils.json_iterator(arguments['INPUT_FILE']):
            docid = obj['id']
            jsonfile = os.path.join(arguments['OUTPUT_DIR'], docid + '.json')
            if not os.path.exists(jsonfile):
                logging.warning('Cannot find file %s', jsonfile)
            else:
                future_map[executor.submit(one_file, jsonfile)] = jsonfile

        for future in futures.as_completed(future_map):
            srcfile = future_map[future]
            try:
                future.result()
            except Exception as exc:
                logging.exception('%s generated an exception: %s', srcfile, exc)


if __name__ == '__main__':
    main_batch(sys.argv[1:])
