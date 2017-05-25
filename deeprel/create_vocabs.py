"""
Usage: 
    create_vocabs.py [options] -o OUTPUT_DIR JSON_DIR INPUT_FILE...

Options:
    --log <str>     Log option. One of DEBUG, INFO, WARNING, ERROR, and CRITICAL. [default: INFO]
    -o OUTPUT_FILE  Output file
    -w WORD2VEC     word2vec file [default: ~/data/word2vec/PubMed-and-PMC-w2v.bin]
    -e              embeddings [default: False]
"""

import json
import logging
import sys

import docopt
import os

from deeprel.model import re_vocabulary
from deeprel.preprocessor import embedding
from deeprel import utils


class VocabsCreater(object):

    vocab_file = 'vocabs.json'
    word_embedding_file = 'word2vec.npz'
    pos_embedding_file = 'pos.npz'
    chunk_embedding_file = 'chunk.npz'
    arg1_dis_embedding_file = 'arg1_dis.npz'
    arg2_dis_embedding_file = 'arg2_dis.npz'
    type_embedding_file = 'type.npz'
    dependency_embedding_file = 'dependency.npz'
    
    def __init__(self, save_path):
        self.save_path = save_path
        self.vocab = re_vocabulary.ReVocabulary()
        self.labels = set()

    def save_vocab(self):
        self.vocab.set_labels(sorted(self.labels))
        self.vocab.freeze()
        re_vocabulary.dump(os.path.join(self.save_path, VocabsCreater.vocab_file), self.vocab)

    def add(self, obj):
        for ex in obj['examples']:
            self.vocab.fit(ex['toks'])
        for rel in obj['relations']:
            self.labels.add(rel['label'])

    def save_word_embeddings(self, src, type):
        dst = os.path.join(self.save_path, VocabsCreater.word_embedding_file)
        embedding.get_word_embeddings(src, self.vocab.vocabs['word'], dst, type)

    def save_pos_embeddings(self):
        dst = os.path.join(self.save_path, VocabsCreater.pos_embedding_file)
        embedding.get_pos_embeddings(self.vocab.vocabs['pos'], dst)

    def save_chunk_embeddings(self):
        dst = os.path.join(self.save_path, VocabsCreater.chunk_embedding_file)
        embedding.get_one_hot(self.vocab.vocabs['chunk'], dst, 'chunk')

    def save_arg1_dis_embeddings(self):
        dst = os.path.join(self.save_path, VocabsCreater.arg1_dis_embedding_file)
        embedding.get_distance_embeddings(self.vocab.vocabs['arg1_dis'], dst, 'c_dis')

    def save_arg2_dis_embeddings(self):
        dst = os.path.join(self.save_path, VocabsCreater.arg2_dis_embedding_file)
        embedding.get_distance_embeddings(self.vocab.vocabs['arg2_dis'], dst, 'd_dis')

    def save_type_embeddings(self):
        dst = os.path.join(self.save_path, VocabsCreater.type_embedding_file)
        embedding.get_one_hot(self.vocab.vocabs['type'], dst, 'type')

    def save_dependency_embedding(self):
        dst = os.path.join(self.save_path, VocabsCreater.dependency_embedding_file)
        embedding.get_one_hot(self.vocab.vocabs['dependency'], dst, 'dependency')


def main(argv):
    arguments = docopt.docopt(__doc__, argv=argv)
    print('create_vocabs')
    print(arguments)
    logging.basicConfig(level=getattr(logging, arguments['--log']), format='%(message)s')

    vc = VocabsCreater(os.path.join(arguments['-o']))

    for obj in utils.json_iterator(arguments['INPUT_FILE']):
        docid = obj['id']
        jsonfile = os.path.join(arguments['JSON_DIR'], docid + '.json')
        if not os.path.exists(jsonfile):
            logging.warning('Cannot find file %s', jsonfile)
        else:
            with open(jsonfile) as jfp:
                obj = json.load(jfp)
                vc.add(obj)
    vc.save_vocab()
    if arguments['-e']:
        if '-w' in arguments:
            vc.save_word_embeddings(arguments['-w'], 'google')
        vc.save_pos_embeddings()
        vc.save_chunk_embeddings()
        vc.save_arg1_dis_embeddings()
        vc.save_arg2_dis_embeddings()
        vc.save_type_embeddings()
        vc.save_dependency_embedding()


if __name__ == '__main__':
    main(sys.argv[1:])
