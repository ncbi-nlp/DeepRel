"""
Usage: 
    create_features.py [options] --output=<directory> INPUT_FILE...

Options:
    --verbose
    --asyn                  asynchronous execution
    --output=<directory>
    -k                      Skip pre-parsed documents [default: False]
"""

import collections
import json
import logging
import math
import os
from concurrent import futures
from pathlib import Path
from subprocess import call

import tqdm

from cli_utils import parse_args
from deeprel import utils
from deeprel.preprocessor import feature


# def one_file(source):
#     logging.debug('Process: %s', source)
#     with open(source) as fp:
#         obj = json.load(fp, object_pairs_hook=collections.OrderedDict)
#     obj['examples'] = feature.generate(obj)
#     with open(source, 'w') as fp:
#         json.dump(obj, fp, indent=2)
from utils import get_max_workers


def syn_process(argv):
    output_dir = Path(argv['--output'])
    for obj in utils.json_iterator(argv['INPUT_FILE']):
        docid = obj['id']
        source = output_dir / (docid + '.json')
        if not source.exists():
            logging.warning('Cannot find file %s', source)
            continue
        try:
            logging.debug('Process: %s', source)
            with open(source) as fp:
                obj = json.load(fp, object_pairs_hook=collections.OrderedDict)
            if argv['-k'] and 'examples' in obj:
                continue
            obj['examples'] = feature.generate(obj)
            with open(source, 'w') as fp:
                json.dump(obj, fp, indent=2)
        except:
            logging.exception('%s generated an exception', source)


def asyn_process(argv):
    max_works = get_max_workers(8)
    print('Max workers', max_works)
    with futures.ProcessPoolExecutor(max_workers=max_works) as executor:
        future_map = {}

        for input_file in argv['INPUT_FILE']:
            with open(input_file) as fp:
                objs = json.load(fp, object_pairs_hook=collections.OrderedDict)

            chunk_size = math.ceil(len(objs) / max_works)
            print('Chunk size', chunk_size)
            for subobjs in utils.chunks(objs, chunk_size):
                tmpfilename = utils.create_tempfile('.json')
                with open(tmpfilename, 'w') as fw:
                    json.dump(subobjs, fw, indent=2)
                cmd = 'python deeprel/create_features.py -k --output {} {}'.format(argv['--output'], tmpfilename)
                print(cmd)
                future = executor.submit(call, cmd.split(' '))
                future_map[future] = cmd

        done_iter = futures.as_completed(future_map)
        if argv['--verbose']:
            done_iter = tqdm.tqdm(done_iter)
        for future in done_iter:
            cmd = future_map[future]
            try:
                future.result()
            except:
                logging.exception(cmd)


if __name__ == '__main__':
    argv = parse_args(__doc__)
    if argv['--asyn']:
        asyn_process(argv)
    else:
        syn_process(argv)
