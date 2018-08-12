"""
Usage: 
    create_vocabs.py [options] --output=<directory> --all=<directory> INPUT_FILE...

Options:
    --verbose
    --output=<directory>  Output file
    --word2vec=<file>     word2vec file
    --embeddings          embeddings [default: False]
    --all=<directory>
"""

import json
import logging
import os
from pathlib import Path

from cli_utils import parse_args
from deeprel import utils
from deeprel.model import re_vocabulary
from deeprel.preprocessor import embedding


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

    def save_word_embeddings(self, src):
        dst = os.path.join(self.save_path, VocabsCreater.word_embedding_file)
        embedding.get_word_embeddings(src, self.vocab.vocabs['word'], dst)

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


if __name__ == '__main__':
    argv = parse_args(__doc__)
    vc = VocabsCreater(os.path.join(argv['--output']))

    json_dir = Path(argv['--all'])
    for obj in utils.json_iterator(argv['INPUT_FILE']):
        docid = obj['id']
        source = json_dir / (docid + '.json')
        if not source.exists():
            logging.warning('Cannot find file %s', source)
        else:
            with open(source) as fp:
                obj = json.load(fp)
                vc.add(obj)
    vc.save_vocab()
    if argv['--embeddings']:
        if '--word2vec' in argv:
            vc.save_word_embeddings(argv['--word2vec'])
        vc.save_pos_embeddings()
        vc.save_chunk_embeddings()
        vc.save_arg1_dis_embeddings()
        vc.save_arg2_dis_embeddings()
        vc.save_type_embeddings()
        vc.save_dependency_embedding()
