from deeprel import nlp


def test_parse():
    b = nlp.Bllip()
    t = b.parse('hello world!')
    assert str(t) == '(S1 (S (NP (NN hello) (NN world) (NN !))))'

if __name__ == '__main__':
    test_parse()
