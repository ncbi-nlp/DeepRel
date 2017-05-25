from deeprel import nlp


def test_parse():
    b = nlp.StanfordParser(['/home/pengy6/panfs/software/stanford-corenlp-full-2016-10-31/*'])
    t = next(b.parse('hello world!'))
    assert str(t) == '(ROOT (S (VP (NP (INTJ (UH hello)) (NP (NN world)))) (. !)))'

if __name__ == '__main__':
    test_parse()
