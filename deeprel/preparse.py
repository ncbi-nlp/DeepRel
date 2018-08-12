"""
Usage: 
    preparse.py [options] --genia=<bin_path> --corenlp=<path> --output=<directory> INPUT_FILE...

Options:
    --verbose
    --genia=<bin_path>
    --corenlp=<path>
    --output=<directory>
    -k              Skip pre-parsed documents [default: False]
"""

import collections
import json
import logging
import os
import sys
from concurrent import futures

import docopt
import tqdm

from cli_utils import parse_args
from deeprel import utils
from deeprel.nlp import BllipParser
from deeprel.nlp import GeniaTagger
from deeprel.nlp import NltkSSplitter
from deeprel.preprocessor import dependency_adder
from deeprel.preprocessor import token_splitter


def get_concepts(mentions):
    """
    Return concept ids
    """
    counter = collections.Counter()
    for ann in mentions:
        counter[ann['id']] += 1
    return [collections.OrderedDict({'id': id, 'freq': freq})
            for id, freq in counter.items()]


class PreParse(object):

    def __init__(self, save_path, genia_path, corenlp_path, skip_preparsed=False):
        """
        Args:
            save_path: output dir
            genia_path: Genia tagger path
        """
        self.save_path = save_path
        self.tagger = GeniaTagger(genia_path)
        self.splitter = NltkSSplitter()
        self.parser = BllipParser(None)
        self.dep_adder = dependency_adder.DependencyAdder()
        self.skip_preparsed = skip_preparsed

    def parse_by_genia(self, text):
        """
        Tokenize and parse the text
        """
        toks = []
        for senttext, offset in self.splitter.split_s(text):
            toks += self.tagger.parse(senttext, offset)
        return toks

    def add_dependency(self, obj):
        self.dep_adder.add_dependency(obj)

    def parse(self, text):
        try:
            return self.parser.parse(text)
        except:
            return None

    def process(self, obj):
        filename = os.path.join(self.save_path, obj['id'] + '.json')
        if self.skip_preparsed and os.path.exists(filename):
            return
        # tokenize
        if 'toks' not in obj:
            obj['toks'] = self.parse_by_genia(obj['text'])
            # re-tokenize
            token_splitter.split(obj['toks'], obj['annotations'])
        # parse
        if 'parse tree' not in obj or obj['parse tree'] is None:
            obj['parse tree'] = self.parse(obj['text'])
            if obj['parse tree'] is not None:
                self.add_dependency(obj)
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=2)


def main_batch(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_map = {}
        for input_file in arguments['INPUT_FILE']:
            with open(input_file) as fp:
                objs = json.load(fp, object_pairs_hook=collections.OrderedDict)

            for subobjs in utils.chunks(objs, 100):
                tmpfilename = utils.create_tempfile('.json')
                with open(tmpfilename, 'w') as fw:
                    json.dump(subobjs, fw, indent=2)
                cmd = 'python deeprel/preparse.py -k --genia {} --corenlp {} --output {}'.format(
                    argv['--genia'], argv['--corenlp'], argv['--output'], tmpfilename)
                argvx = docopt.docopt(__doc__, cmd.split(' '))
                future = executor.submit(syn_process, argvx)
                future_map[future] = argvx
        for future in futures.as_completed(future_map):
            argvx = future_map[future]
            try:
                future.result()
            except Exception as exc:
                print('{} generated an exception: {}'.format(argvx, exc))


def syn_process(argv):
    preparse = PreParse(argv['--output'], argv['--genia'], [argv['--corenlp']], argv['-k'])
    for obj in utils.json_iterator(argv['INPUT_FILE']):
        preparse.process(obj)


if __name__ == '__main__':
    argv = parse_args(__doc__)
    if argv['--corenlp'][-1] == '/':
        argv['--corenlp'] += '*'
    else:
        argv['--corenlp'] += '/*'

    syn_process(argv)
