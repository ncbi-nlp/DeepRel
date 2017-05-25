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


def test_locations(section):
    jsondir = os.path.join(section['model_dir'], 'all')
    if not os.path.exists(jsondir):
        os.makedirs(jsondir)

    training_set = os.path.join(section['model_dir'], section['training_set'] + '.json')
    test_set = os.path.join(section['model_dir'], section['test_set'] + '.json')
    dev_set = os.path.join(section['model_dir'], section['dev_set'] + '.json')

    assert os.path.exists(training_set)
    assert os.path.exists(test_set)
    assert os.path.exists(dev_set)
    assert os.path.exists(section['geniatagger'])
    assert os.path.exists(section['corenlp_jars'])
    assert os.path.exists(section['word2vec'])


def main():
    arguments = docopt.docopt(__doc__)
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    config = configparser.ConfigParser()
    config.read(arguments['INI_FILE'])

    cnn_section = config['cnn']
    test_locations(cnn_section)

    jsondir = os.path.join(cnn_section['model_dir'], 'all')
    training_set = str(os.path.join(cnn_section['model_dir'], cnn_section['training_set'] + '.json'))
    test_set = str(os.path.join(cnn_section['model_dir'], cnn_section['test_set'] + '.json'))
    dev_set = str(os.path.join(cnn_section['model_dir'], cnn_section['dev_set'] + '.json'))

    if arguments['-p']:
        argv = [
            cnn_section['geniatagger'],
            cnn_section['corenlp_jars'],
            jsondir,
            training_set,
            test_set,
            dev_set,
        ]
        preparse.main(argv)
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
            '-w', cnn_section['word2vec'],
            '-e',
            '-o', cnn_section['model_dir'],
            jsondir,
            training_set,
            test_set,
            dev_set,
        ]
        create_vocabs.main(argv)
    if arguments['-m']:
        vocab = os.path.join(cnn_section['model_dir'], VocabsCreater.vocab_file)
        for src, dst in zip([training_set, test_set, dev_set], ['training_set', 'test_set', 'dev_set']):
            argv = [
                vocab,
                jsondir,
                src,
                os.path.join(cnn_section['model_dir'], cnn_section[dst] + '.npz')
            ]
            create_matrix.main(argv)
    if arguments['-s']:
        vocab = os.path.join(cnn_section['model_dir'], VocabsCreater.vocab_file)
        for src, dst in zip([training_set, test_set, dev_set], ['training_set', 'test_set', 'dev_set']):
            argv = [
                vocab,
                jsondir,
                src,
                os.path.join(cnn_section['model_dir'], cnn_section[dst] + '-sp.npz')
            ]
            create_sp_matrix.main(argv)
    if arguments['-t']:
        argv = [
            os.path.join(cnn_section['model_dir'], VocabsCreater.vocab_file),
            os.path.join(cnn_section['model_dir'], cnn_section['training_set'] + '.npz'),
            os.path.join(cnn_section['model_dir'], cnn_section['test_set'] + '.npz'),
            os.path.join(cnn_section['model_dir'], cnn_section['dev_set'] + '.npz')
        ]
        test_matrix.main(argv)
        argv = [
            os.path.join(cnn_section['model_dir'], VocabsCreater.vocab_file),
            os.path.join(cnn_section['model_dir'], cnn_section['training_set'] + '-sp.npz'),
            os.path.join(cnn_section['model_dir'], cnn_section['test_set'] + '-sp.npz'),
            os.path.join(cnn_section['model_dir'], cnn_section['dev_set'] + '-sp.npz')
        ]
        test_matrix.main(argv)
    if arguments['-d']:
        model_file = os.path.join(cnn_section['model_dir'], cnn_section['training_set'] + '.doc2vec')
        argv = [
            jsondir,
            training_set,
            model_file
        ]
        train_doc2vec.main(argv)
        for src, dst in zip([training_set, test_set, dev_set], ['training_set', 'test_set', 'dev_set']):
            argv = [
                jsondir,
                model_file,
                src,
                os.path.join(cnn_section['model_dir'], cnn_section[dst] + '-doc.npz'),
            ]
            create_doc2vec.main(argv)


if __name__ == '__main__':
    main()