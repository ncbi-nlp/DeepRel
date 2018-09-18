"""
Usage:
    run.py [options] INI_FILE
    
Options:
    --verbose
    -p           preparse [default: False]
    -f           create features [default: False]
    -v           create vocabularies [default: False]
    -m           create matrix [default: False]
    -s           create shortest path matrix [default: False]
    -d           create doc2vec [default: False]
    -u           create universal sentence [default: False]
    -t           test matrix format [default: False]
    -k           skip pre-parsed documents [default: False]
"""
import configparser
from collections import namedtuple
from pathlib import Path
from subprocess import call

import numpy as np

import test_matrix
from cli_utils import parse_args
from deeprel.create_vocabs import VocabsCreater


class Locations2(object):
    def __init__(self, section):
        self.section = section

    @property
    def model_dir(self):
        return Path(self.section['model_dir'])

    @property
    def jsondir(self):
        return self.model_dir / 'all'

    @property
    def geniatagger(self):
        return Path(self.section['geniatagger'])

    @property
    def corenlp_jars(self):
        return Path(self.section['corenlp_jars'])

    @property
    def word2vec(self):
        return Path(self.section['word2vec'])

    @property
    def vocab(self):
        return self.model_dir / VocabsCreater.vocab_file

    def dataset(self, name):
        return self.model_dir / (self.section[name] + '.json')

    def npz(self, name):
        return self.model_dir / (self.section[name] + '.npz')

    def sp_npz(self, name):
        return self.model_dir / (self.section[name] + '-sp.npz')

    def doc_npz(self, name):
        return self.model_dir / (self.section[name] + '-doc.npz')

    def universal_npz(self, name):
        return self.model_dir / (self.section[name] + '-uni.npz')

    def datasets(self):
        return [self.dataset(n) for n in ('training_set', 'test_set', 'dev_set')]


def test_locations(section):
    l = Locations2(section)

    if not l.jsondir.exists():
        l.jsondir.mkdir(parents=True)

    assert l.dataset('training_set').exists()
    assert l.dataset('test_set').exists()
    assert l.dataset('dev_set').exists()
    assert l.geniatagger.exists()
    assert l.corenlp_jars.exists()
    assert l.word2vec.exists()

    return l


if __name__ == '__main__':
    arguments = parse_args(__doc__)

    config = configparser.ConfigParser()
    config.read(arguments['INI_FILE'])

    cnn_section = config['cnn']
    l = test_locations(cnn_section)

    if arguments['-p']:
        cmd = 'python deeprel/preparse.py --asyn -k --genia {} --corenlp {} --output {} {} {} {}'.format(
            l.geniatagger, l.corenlp_jars, l.jsondir, *l.datasets())
        print(cmd)
        call(cmd.split(' '))
    if arguments['-f']:
        cmd = 'python deeprel/create_features.py --asyn -k --output {} {} {} {}'.format(
            l.jsondir, *l.datasets())
        print(cmd)
        call(cmd.split(' '))
    if arguments['-v']:
        cmd = 'python deeprel/create_vocabs.py --word2vec {} --embeddings --output {} --all {} {} {} {}'.format(
            l.word2vec, l.model_dir, l.jsondir, *l.datasets())
        print(cmd)
        call(cmd.split(' '))
    if arguments['-m']:
        cmd_prefix = 'python deeprel/create_matrix.py matrix --vocab {} --all {}'.format(l.vocab, l.jsondir)
        for name in ('training_set', 'test_set', 'dev_set'):
            if not arguments['-k'] or not l.npz(name).exists():
                cmd = cmd_prefix + ' --output {} {}'.format(l.npz(name), l.dataset(name))
                print(cmd)
                call(cmd.split(' '))
    if arguments['-s']:
        cmd_prefix = 'python deeprel/create_matrix.py sp --vocab {} --all {}'.format(l.vocab, l.jsondir)
        for name in ('training_set', 'test_set', 'dev_set'):
            if not arguments['-k'] or not l.sp_npz(name).exists():
                cmd = cmd_prefix + ' --output {} {}'.format(l.sp_npz(name), l.dataset(name))
                print(cmd)
                call(cmd.split(' '))
    if arguments['-d']:
        model_file = l.model_dir / (cnn_section['training_set'] + '.doc2vec')
        if not arguments['-k'] or not model_file.exists():
            cmd = 'python deeprel/doc2vec.py fit --all {} --model {} {} {} {}'.format(
                l.jsondir, model_file, l.dataset('training_set'), l.dataset('dev_set'), l.dataset('test_set'))
            print(cmd)
            call(cmd.split(' '))

        cmd_prefix = 'python deeprel/doc2vec.py transform --verbose --all {} --model {}'.format(l.jsondir, model_file)
        for name in ('training_set', 'test_set', 'dev_set'):
            if not arguments['-k'] or not l.doc_npz(name).exists():
                cmd = cmd_prefix + ' --output {} {}'.format(l.doc_npz(name), l.dataset(name))
                print(cmd)
                call(cmd.split(' '))
    if arguments['-u']:
        cmd_prefix = 'python deeprel/universal_sentence.py transform --verbose --all {}'.format(l.jsondir)
        for name in ('training_set', 'test_set', 'dev_set'):
            if not arguments['-k'] or not l.doc_npz(name).exists():
                cmd = cmd_prefix + ' --output {} {}'.format(l.universal_npz(name), l.dataset(name))
                print(cmd)
                call(cmd.split(' '))
    if arguments['-t']:
        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.npz('training_set'), l.npz('test_set'))
        call(cmd.split(' '))

        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.npz('training_set'), l.npz('dev_set'))
        call(cmd.split(' '))

        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.sp_npz('training_set'), l.sp_npz('test_set'))
        call(cmd.split(' '))

        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.sp_npz('training_set'), l.sp_npz('dev_set'))
        call(cmd.split(' '))

        for name in ('training_set', 'test_set', 'dev_set'):
            test_matrix.test_doc(l.npz(name), l.doc_npz(name))
            test_matrix.test_universal(l.npz(name), l.universal_npz(name))
