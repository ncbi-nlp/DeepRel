"""
Usage: 
    preparse.py [options] --genia=<bin_path> --input=<file> --output=<file>

Options:
    --verbose=<int>     logging level. [default: 0]
    --genia=<bin_path>  Genia tagger binary
"""
import copy
from functools import partial

from deeprel.nlp import BllipParser, Ptb2Dep
from deeprel.nlp import GeniaTagger
from nlp.nltk_ssplit import split_text
from utils import precess_jsonl, parse_args


def parse_by_genia(text, genia_tagger):
    """
    Tokenize and parse the text
    """
    toks = []
    for senttext, offset in split_text(text):
        toks += genia_tagger.parse(senttext, offset)
    return toks


def preparse(obj, genia_tagger, bllip_parser, ptb2dep):
    # tokenize
    if 'toks' not in obj:
        obj['toks'] = parse_by_genia(obj['text'], genia_tagger)
        # re-tokenize
        re_tokenize(obj['toks'], obj['annotations'])
    # parse
    if 'parse tree' not in obj or obj['parse tree'] is None:
        try:
            obj['parse tree'] = bllip_parser.parse(obj['text'])
        except:
            obj['parse tree'] = None
        if obj['parse tree'] is not None:
            ptb2dep.add_dependency(obj)
    return obj


def re_tokenize(tokens, mentions):

    for mention in mentions:
        # find toks
        for i, tok in enumerate(tokens):
            if tok['start'] == mention['start'] and tok['end'] == mention['end']:
                break
            # [tok/mention ]mention ]tok
            if tok['start'] == mention['start'] and mention['end'] < tok['end']:
                pivot = mention['end']-tok['start']
                tok1 = copy.deepcopy(tok)
                tok1['word'] = tok['word'][:pivot]
                tok1['base'] = tok['base'][:pivot]
                tok1['end'] = mention['end']

                tok2 = copy.deepcopy(tok)
                tok2['word'] = tok['word'][pivot:]
                tok2['base'] = tok['base'][pivot:]
                tok2['start'] = mention['end']

                if tok['chunk'][0] == 'B':
                    tok2['chunk'] = 'I' + tok['chunk'][1:]
                elif tok['chunk'] == 'O':
                    tok1['chunk'] = 'B-NP'
                if tok['ne'][0] == 'B':
                    tok2['ne'] = 'I' + tok['ne'][1:]
                elif tok['ne'] == 'O':
                    tok1['ne'] = 'B-protein'

                tokens.remove(tok)
                tokens.insert(i, tok1)
                tokens.insert(i + 1, tok2)

                break
            # [tok [mention ]mention/tok
            if tok['start'] < mention['start'] and mention['end'] == tok['end']:
                pivot = mention['start']-tok['start']
                tok1 = copy.deepcopy(tok)
                tok1['word'] = tok['word'][:pivot]
                tok1['base'] = tok['base'][:pivot]
                tok1['end'] = mention['start']

                tok2 = copy.deepcopy(tok)
                tok2['word'] = tok['word'][pivot:]
                tok2['base'] = tok['base'][pivot:]
                tok2['start'] = mention['start']
                if tok['chunk'][0] == 'B':
                    tok2['chunk'] = 'I' + tok['chunk'][1:]
                elif tok['chunk'] == 'O':
                    tok2['chunk'] = 'B-NP'

                if tok['ne'][0] == 'B':
                    tok1['ne'] = 'O'
                elif tok['ne'] == 'O':
                    tok2['ne'] = 'B-protein'

                tokens.remove(tok)
                tokens.insert(i, tok1)
                tokens.insert(i + 1, tok2)
                break


if __name__ == '__main__':
    argv = parse_args(__doc__)
    func = partial(preparse,
                   genia_tagger=GeniaTagger(argv['--genia']),
                   bllip_parser=BllipParser(None),
                   ptb2dep=Ptb2Dep(universal=True))
    precess_jsonl(argv['--input'], argv['--output'], func, verbose=argv['--verbose'] > 0)
