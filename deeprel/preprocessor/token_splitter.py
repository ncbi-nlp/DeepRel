import copy


def split(tokens, mentions):

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