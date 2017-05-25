from deeprel import vocabulary


def test_vocabulary():
    l = ['a', 'b', 'c']
    actual = vocabulary.Vocabulary()
    for c in l:
        actual.add(c)
    actual.freeze(True)

    assert len(actual) == 3
    for i, c in enumerate(l):
        assert actual.get(c) == i
        assert actual.reverse(i) == c

    assert actual.category2index['a'] == 0
    assert actual.index2category[0] == 'a'

    copy = vocabulary.Vocabulary()
    copy.update(l)
    copy.freeze(True)
    assert copy == actual


if __name__ == '__main__':
    test_vocabulary()
