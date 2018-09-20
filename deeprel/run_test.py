"""
Usage:
    run_test.py [options] CONFIG_FILE
    
Options:
    --verbose
    -p           preparse [default: False]
    -f           create features [default: False]
    -m           create matrix [default: False]
    -s           create shortest path matrix [default: False]
    -d           create doc2vec [default: False]
    -t           test matrix format [default: False]
    -k           skip pre-parsed documents [default: False]
"""
import configparser
import json
from subprocess import call

import test_matrix
from utils import parse_args
from run import Locations2


def test_locations(section):
    l = Locations2(section)
    assert l.dataset('test_set').exists()
    assert l.geniatagger.exists()
    assert l.word2vec.exists()

    return l


if __name__ == '__main__':
    arguments = parse_args(__doc__)

    with open(arguments['CONFIG_FILE']) as fp:
        config = json.load(fp)

    l = test_locations(config)

    if arguments['-p']:
        input = l.dataset('test_set')
        output, error = l.preprocessed('test_set')
        cmd = 'python deeprel/preparse.py --genia {} --input {} --output {} '.format(
            l.geniatagger, input, output)
        print(cmd)
        call(cmd, shell=True)
    if arguments['-f']:
        input = l.preprocessed('test_set')
        output, error = l.features('test_set')
        cmd = 'python deeprel/create_features.py --asyn --input {} --output {} --error {}'.format(
            l.geniatagger, input, output, error)
        print(cmd)
        call(cmd, shell=True)
    if arguments['-m']:
        if not arguments['-k'] or not l.npz('test_set').exists():
            cmd = 'python deeprel/create_matrix.py matrix --vocab {} --all {} --output {} {}'.format(
                l.vocab, l.jsondir, l.npz('test_set'), l.dataset('test_set'))
            print(cmd)
            call(cmd.split(' '))
    if arguments['-s']:
        if not arguments['-k'] or not l.sp_npz('test_set').exists():
            cmd = 'python deeprel/create_matrix.py sp --vocab {} --all {} --output {} {}'.format(
                l.vocab, l.jsondir, l.sp_npz('test_set'), l.dataset('test_set'))
            print(cmd)
            call(cmd.split(' '))
    if arguments['-d']:
        model_file = l.model_dir / (cnn_section['training_set'] + '.doc2vec')
        if not arguments['-k'] or not l.doc_npz('test_set').exists():
            cmd = 'python deeprel/doc2vec.py transform --verbose --all {} --model {} --output {} {}'.format(
                l.jsondir, model_file, l.doc_npz('test_set'), l.dataset('test_set'))
            print(cmd)
            call(cmd.split(' '))
    if arguments['-t']:
        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.npz('training_set'), l.npz('test_set'))
        call(cmd.split(' '))

        cmd = 'python deeprel/test_matrix.py --vocab {} {} {}'.format(
            l.vocab, l.sp_npz('training_set'), l.sp_npz('test_set'))
        call(cmd.split(' '))

        test_matrix.test_doc(l.npz('test_set'), l.doc_npz('test_set'))
