"""
Usage:
    run_test.py [options] INI_FILE
    
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
from subprocess import call

import test_matrix
from cli_utils import parse_args
from run import Locations2


def test_locations(section):
    l = Locations2(section)

    if not l.jsondir.exists():
        l.jsondir.mkdir(parents=True)

    assert l.dataset('test_set').exists()
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
        cmd_template = 'python deeprel/preparse.py --asyn --genia {} --corenlp {} --output {} {}'
        if arguments['-k']:
            cmd_template += ' -k'
        cmd = cmd_template.format(l.geniatagger, l.corenlp_jars, l.jsondir, l.dataset('test_set'))
        call(cmd.split(' '))
    if arguments['-f']:
        cmd_template = 'python deeprel/create_features.py --asyn --output {} {}'
        if arguments['-k']:
            cmd_template += ' -k'
        cmd = cmd_template.format(l.jsondir, l.dataset('test_set'))
        call(cmd.split(' '))
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
