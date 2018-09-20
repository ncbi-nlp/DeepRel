"""
Usage:
    run.py [options] CONFIG_FILE
    
Options:
    --verbose=<int>  logging level. [default: 0]
    -i           initialize the folder structure
    -p           preparse [default: False]
    -f           create features [default: False]
    -v           create vocabularies [default: False]
    -m           create matrix [default: False]
    -s           create shortest path matrix [default: False]
    -u           create universal sentence [default: False]
    -t           test matrix format [default: False]
    -k           skip pre-parsed documents [default: False]
    -w           create sent2vec sentence [default: False]
    -c           combine files [default: False]
"""
import json
import logging
import os
from pathlib import Path
from subprocess import call

import tqdm

import test_matrix
from deeprel.create_vocabs import VocabsCreater
from utils import parse_args, submit_cmds, to_path, get_max_workers


class Locations2(object):
    def __init__(self, section):
        self.config = section

    @property
    def model_dir(self):
        return Path(self.config['model_dir'])

    @property
    def geniatagger(self):
        return Path(self.config['geniatagger'])

    @property
    def corenlp_jars(self):
        return Path(self.config['corenlp_jars'])

    @property
    def word2vec(self):
        return self.config['word2vec']

    @property
    def vocab(self):
        return self.model_dir / VocabsCreater.vocab_file

    def dataset(self, name):
        return self.model_dir / (self.config[name] + '.jsonl')

    def npz(self, name):
        return self.model_dir / (self.config[name] + '.npz')

    def sp_npz(self, name):
        return self.model_dir / (self.config[name] + '-sp.npz')

    def doc_npz(self, name):
        return self.model_dir / (self.config[name] + '-doc.npz')

    def universal_npz(self, name):
        return self.model_dir / (self.config[name] + '-uni.npz')

    def sent2vec_npz(self, name):
        return self.model_dir / (self.config[name] + '-s2v.npz')

    def features(self, name):
        return self.model_dir / (self.config[name] + '-features.jsonl')

    def preprocessed(self, name):
        return self.model_dir / (self.config[name] + '-preprocessed.jsonl')

    def folder(self, name):
        source = to_path(self.dataset(name))
        return source.parent / source.stem

    def preprocessed_folder(self, name):
        folder = self.folder(name)
        return folder.parent / (folder.stem + '-preprocessed')

    def features_folder(self, name):
        folder = self.folder(name)
        return folder.parent / (folder.stem + '-features')


def test_locations(section):
    l = Locations2(section)
    assert l.dataset('training_set').exists()
    assert l.dataset('test_set').exists()
    assert l.dataset('dev_set').exists()
    assert l.geniatagger.exists()

    index = l.word2vec.find(':')
    assert Path(l.word2vec[index + 1:]).exists()

    return l


def split_lines(source, number_of_lines=100):
    source = to_path(source)
    dest_dir = source.parent / source.stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = 'split --additional-suffix=.jsonl -l {} {} {}'.format(number_of_lines, source, str(dest_dir) + '/')
    logging.info(cmd)
    call(cmd, shell=True, cwd=str(dest_dir))


def equal_lines(file1, file2):
    count1 = 0
    count2 = 0
    for _ in open(file1): count1 += 1
    for _ in open(file2): count2 += 1
    return count1 == count2


def get_cmds(cmd, input_dir, output_dir, verbose=True):
    cmds = []
    with os.scandir(input_dir) as it:
        if verbose:
            it = tqdm.tqdm(it, unit='files')
        for entry in it:
            if 'errorlines' in entry.name:
                continue
            input = Path(entry.path)
            output = output_dir / input.name
            if not output.exists() or not equal_lines(input, output):
                entry_cmd = '{} --input {} --output {}'.format(cmd, input, output)
                cmds.append(entry_cmd)
    return cmds


if __name__ == '__main__':
    argv = parse_args(__doc__)

    with open(argv['CONFIG_FILE']) as fp:
        config = json.load(fp)

    l = test_locations(config)

    if argv['-i']:
        # prepare the folder
        for name in ('training_set', 'test_set', 'dev_set'):
            split_lines(l.dataset(name))

    if argv['-p']:
        cmd_prefix = 'python deeprel/preparse.py --verbose {} --genia {}'.format(argv['--verbose'], l.geniatagger)
        cmds = []
        for name in ('training_set', 'test_set', 'dev_set'):
            l.preprocessed_folder(name).mkdir(parents=True, exist_ok=True)
            cmds += get_cmds(cmd_prefix, l.folder(name), l.preprocessed_folder(name))
        submit_cmds(cmds)

    if argv['-f']:
        cmd_prefix = 'python deeprel/create_features.py --verbose {}'.format(argv['--verbose'])
        cmds = []
        for name in ('training_set', 'test_set', 'dev_set'):
            l.features_folder(name).mkdir(parents=True, exist_ok=True)
            cmds += get_cmds(cmd_prefix, l.preprocessed_folder(name), l.features_folder(name))
        submit_cmds(cmds)

    if argv['-c']:
        # combine
        cmds = []
        for name in ('training_set', 'test_set', 'dev_set'):
            cmd = 'cat'
            with os.scandir(l.features_folder(name)) as it:
                if argv['--verbose'] > 0:
                    it = tqdm.tqdm(it, unit='files')
                for entry in it:
                    if 'errorlines' in entry.name:
                        continue
                    cmd += " '{}'".format(entry.path)
            cmd += ' > {}'.format(l.features(name))
            cmds.append(cmd)
        submit_cmds(cmds)

    if argv['-v']:
        cmd = 'python deeprel/create_vocabs.py --verbose --word2vec {} --embeddings --output {} '.format(
            l.word2vec, l.model_dir)
        cmd += ' '.join(str(l.features(name)) for name in ('training_set', 'test_set', 'dev_set'))
        print(cmd)
        call(cmd, shell=True)

    if argv['-m']:
        cmd_prefix = 'python deeprel/create_matrix.py matrix --verbose --vocab {}'.format(l.vocab)
        cmds = []
        for name in ('training_set', 'test_set', 'dev_set'):
            if not argv['-k'] or not l.npz(name).exists():
                cmd = cmd_prefix + ' --input {} --output {}'.format(l.features(name), l.npz(name))
                cmds.append(cmd)
        submit_cmds(cmds)

    if argv['-s']:
        cmd_prefix = 'python deeprel/create_matrix.py sp --verbose --vocab {}'.format(l.vocab)
        cmds = []
        for name in ('training_set', 'test_set', 'dev_set'):
            if not argv['-k'] or not l.sp_npz(name).exists():
                cmd = cmd_prefix + ' --input {} --output {}'.format(l.features(name), l.sp_npz(name))
                cmds.append(cmd)
        submit_cmds(cmds)

    if argv['-u']:
        cmd_prefix = 'python deeprel/universal_sentence.py --verbose'
        cmds = []
        for name in ('training_set', 'test_set', 'dev_set'):
            if not argv['-k'] or not l.universal_npz(name).exists():
                cmd = cmd_prefix + ' --output {} --input {}'.format(l.universal_npz(name), l.features(name))
                cmds.append(cmd)
        submit_cmds(cmds)

    if argv['-w']:
        cmd_prefix = 'python deeprel/sent2vec_sentence.py --verbose'
        cmds = []
        for name in ('training_set', 'test_set', 'dev_set'):
            if not argv['-k'] or not l.sent2vec_npz(name).exists():
                cmd = cmd_prefix + ' --output {} --input {}'.format(l.sent2vec_npz(name), l.features(name))
                cmds.append(cmd)
        submit_cmds(cmds)

    if argv['-t']:
        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.npz('training_set'), l.npz('test_set'))
        call(cmd, shell=True)

        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.npz('training_set'), l.npz('dev_set'))
        call(cmd, shell=True)

        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.sp_npz('training_set'), l.sp_npz('test_set'))
        call(cmd, shell=True)

        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.sp_npz('training_set'), l.sp_npz('dev_set'))
        call(cmd, shell=True)

        for name in ('training_set', 'test_set', 'dev_set'):
            # test_matrix.test_doc(l.npz(name), l.doc_npz(name))
            # test_matrix.test_universal(l.npz(name), l.universal_npz(name))
            test_matrix.test_s2v(l.npz(name), l.sent2vec_npz(name))
