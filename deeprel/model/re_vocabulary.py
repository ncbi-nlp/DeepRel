import json
import logging

from deeprel import vocabulary


class ReVocabulary(object):
    """
    Contains everything needed for relation extraction except for the model
    """

    def __init__(self, keys=None):
        if keys:
            self.keys = keys
        else:
            self.keys = ['word', 'pos', 'chunk', 'arg1_dis', 'arg2_dis', 'type', 'dependency']
        self.vocabs = {
            k: vocabulary.Vocabulary() for k in self.keys
        }
        self.label_vocab = vocabulary.Vocabulary(unknown=False)
        self.max_len = 0

    def __getitem__(self, item):
        return self.vocabs[item]

    def set_labels(self, labels):
        self.label_vocab.update(labels)

    def __eq__(self, other):
        if not isinstance(other, ReVocabulary):
            return False
        return self.keys == other.keys \
               and self.vocabs == other.vocabs \
               and self.max_len == other.max_len

    def fit(self, toks):
        """Learn vocabulary dictionaries of all tokens in the examples"""
        for tok in toks:
            for k in self.keys:
                try:
                    self.vocabs[k].add(str(tok[k]))
                except:
                    logging.debug('Cannot find key %s in %s', k, tok)
        self.max_len = max(self.max_len, len(toks))

    def freeze(self):
        for k in self.keys:
            self.vocabs[k].freeze()

    def summary(self):
        max_key_width = len('dependency')
        for k in self.keys:
            print('{} vocab, number of labels: {}'.format(
                k.ljust(max_key_width), len(self.vocabs[k])))
        print('{} vocab, number of labels: {}'.format(
            'label'.ljust(max_key_width), len(self.label_vocab)))
        print('Max length: {}'.format(self.max_len))


def load(filename):
    """
    Restores vocabulary processor from given file.

    Args:
      filename: Path to file to load from.
    Returns:
      ReVocabulary object.
    """
    with open(filename) as fp:
        return json.load(fp, cls=ReVocabularyDecoder)


def dump(filename, vocab):
    """Saves vocabulary processor into given file.

    Args:
      filename: Path to output file.
    """
    with open(filename, 'w') as f:
        json.dump(vocab, f, indent=2, cls=ReVocabularyEncoder)


class ReVocabularyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ReVocabulary):
            return {
                "keys": obj.keys,
                "vocabs": {key: obj.vocabs[key].__dict__ for key in obj.keys},
                "label_vocab": obj.label_vocab.__dict__,
                "max_len": obj.max_len
            }
        return json.JSONEncoder.default(self, obj)


class ReVocabularyDecoder(json.JSONDecoder):
    def decode(self, s, _w=None):
        d = super(ReVocabularyDecoder, self).decode(s)
        voc = ReVocabulary(keys=d['keys'])
        voc.vocabs = {key: self._decode_vocab(d['vocabs'][key]) for key in d['vocabs']}
        voc.label_vocab = self._decode_vocab(d['label_vocab'])
        voc.max_len = int(d['max_len'])
        return voc

    @classmethod
    def _decode_vocab(cls, obj):
        v = vocabulary.Vocabulary()
        for k in obj:
            setattr(v, k, obj[k])
        return v
