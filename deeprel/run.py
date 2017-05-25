"""
Usage:
    run.py [options] INI_FILE
    
Options:
    --log <str>  Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
    -p           preparse [default: False]
    -f           create features [default: False]
    -v           create vocabularies [default: False]
    -m           create matrix [default: False]
    -s           create shortest path matrix [default: False]
    -t           test matrix [default: False]
    -d           create doc2vec [default: False]
"""
import docopt
import logging
import configparser
import os

from deeprel import preparse
from deeprel import create_features
from deeprel import create_vocabs
from deeprel import create_matrix
from deeprel import create_sp_matrix
from deeprel import test_matrix
from deeprel import create_doc2vec
from deeprel import train_doc2vec
from deeprel.create_vocabs import VocabsCreater


def test_locations(config):
    jsondir = os.path.join(config['model_dir'], 'all')
    if not os.path.exists(jsondir):
        os.makedirs(jsondir)

    training_set = os.path.join(config['model_dir'], config['training_set'] + '.json')
    test_set = os.path.join(config['model_dir'], config['test_set'] + '.json')
    dev_set = os.path.join(config['model_dir'], config['dev_set'] + '.json')

    assert os.path.exists(training_set)
    assert os.path.exists(test_set)
    assert os.path.exists(dev_set)
    assert os.path.exists(config['geniatagger'])
    assert os.path.exists(config['corenlp_jars'])
    assert os.path.exists(config['word2vec'])


def main():
    arguments = docopt.docopt(__doc__)
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    config = configparser.ConfigParser()
    config.read(arguments['INI_FILE'])

    test_locations(config)
    jsondir = os.path.join(config['model_dir'], 'all')
    training_set = os.path.join(config['model_dir'], config['training_set'] + '.json')
    test_set = os.path.join(config['model_dir'], config['test_set'] + '.json')
    dev_set = os.path.join(config['model_dir'], config['dev_set'] + '.json')

    if arguments['-p']:
        argv = [
            config['geniatagger'],
            config['corenlp_jars'],
            jsondir,
            training_set,
            test_set,
            dev_set,
        ]
        preparse.main_batch(argv)
    if arguments['-f']:
        argv = [
            jsondir,
            training_set,
            test_set,
            dev_set,
        ]
        create_features.main_batch(argv)
    if arguments['-v']:
        argv = [
            '-w', config['word2vec'],
            '-e',
            '-o', config['model_dir'],
            jsondir,
            training_set,
            test_set,
            dev_set,
        ]
        create_vocabs.main(argv)
    if arguments['-m']:
        argv = [
            os.path.join(config['model_dir'], VocabsCreater.vocab_file),
            jsondir,
            training_set,
            os.path.join(config['model_dir'], config['training_set'] + '.npz')
        ]
        create_matrix.main(argv)
        argv = [
            os.path.join(config['model_dir'], VocabsCreater.vocab_file),
            jsondir,
            training_set,
            os.path.join(config['model_dir'], config['test_set'] + '.npz')
        ]
        create_matrix.main(argv)
        argv = [
            os.path.join(config['model_dir'], VocabsCreater.vocab_file),
            jsondir,
            training_set,
            os.path.join(config['model_dir'], config['dev_set'] + '.npz')
        ]
        create_matrix.main(argv)
    if arguments['-s']:
        argv = [
            os.path.join(config['model_dir'], VocabsCreater.vocab_file),
            jsondir,
            training_set,
            os.path.join(config['model_dir'], config['training_set'] + '-sp.npz')
        ]
        create_sp_matrix.main(argv)
        argv = [
            os.path.join(config['model_dir'], VocabsCreater.vocab_file),
            jsondir,
            training_set,
            os.path.join(config['model_dir'], config['test_set'] + '-sp.npz')
        ]
        create_sp_matrix.main(argv)
        argv = [
            os.path.join(config['model_dir'], VocabsCreater.vocab_file),
            jsondir,
            training_set,
            os.path.join(config['model_dir'], config['dev_set'] + '-sp.npz')
        ]
        create_sp_matrix.main(argv)
    if arguments['-t']:
        argv = [
            os.path.join(config['model_dir'], VocabsCreater.vocab_file),
            os.path.join(config['model_dir'], config['training_set'] + '-sp.npz'),
            os.path.join(config['model_dir'], config['test_set'] + '-sp.npz'),
            os.path.join(config['model_dir'], config['dev_set'] + '-sp.npz')
        ]
        test_matrix.main(argv)
    if arguments['-d']:
        model_file = os.path.join(config['model_dir'], config['training_set'] + '.doc2vec')
        argv = [
            jsondir,
            training_set,
            model_file
        ]
        train_doc2vec.main(argv)
        argv = [
            jsondir,
            model_file,
            training_set,
            os.path.join(config['model_dir'], config['training_set'] + '-doc.npz'),
        ]
        create_doc2vec.main(argv)
        argv = [
            jsondir,
            model_file,
            test_set,
            os.path.join(config['model_dir'], config['test_set'] + '-doc.npz'),
        ]
        create_doc2vec.main(argv)
        argv = [
            jsondir,
            model_file,
            dev_set,
            os.path.join(config['model_dir'], config['dev_set'] + '-doc.npz'),
        ]
        create_doc2vec.main(argv)


if __name__ == '__main__':
    main()