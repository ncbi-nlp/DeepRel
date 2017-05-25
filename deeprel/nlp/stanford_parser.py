from stanford_corenlp_pywrapper import CoreNLP


class StanfordParser(object):
    """
    Stanford parser
    """
    def __init__(self, corenlp_jars):
        self.proc = CoreNLP("parse", corenlp_jars=corenlp_jars)

    def parse(self, text):
        # {u'sentences':
        #     [
        #         {u'parse': u'(ROOT (S (VP (NP (INTJ (UH hello)) (NP (NN world)))) (. !)))'
        #          u'tokens': [u'hello', u'world', u'.'],
        #          u'lemmas': [u'hello', u'world', u'.'],
        #          u'pos': [u'UH', u'NN', u'.'],
        #          u'char_offsets': [[0, 5], [6, 11], [11, 12]]
        #          },
        #         ...
        #     ]
        # }
        json_rst = self.proc.parse_doc(text)
        if json_rst:
            for sent in json_rst['sentences']:
                parse_tree= sent['parse']
                yield parse_tree
