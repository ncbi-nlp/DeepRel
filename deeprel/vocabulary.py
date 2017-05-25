import collections


class Vocabulary(object):
    """Categorical variables vocabulary class.

    Accumulates and provides mapping from classes to indexes.
    Can be easily used for words.
    """
    def __init__(self, categories=None):
        self.mapping = {}
        self.reverse_mapping = []
        self.freq = collections.Counter()
        self.is_frozen = False
        if categories:
            self.update(categories)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, Vocabulary):
            return False
        return self.reverse_mapping == other.reverse_mapping \
            and self.is_frozen == other.is_frozen \
            and self.mapping == other.mapping \
            and self.freq == other.freq

    def __len__(self):
        """Returns total count of mappings. Including unknown token."""
        return len(self.mapping)

    def freeze(self, freeze=True):
        """Freezes the vocabulary, after which new words return unknown token id.

        Args:
            freeze: True to freeze, False to unfreeze.
        """
        self.is_frozen = freeze

    def get(self, category):
        """
        Returns category's index in the vocabulary.

        Args:
            category: string or integer to lookup in vocabulary.

        Returns:
            int: index in the vocabulary.
        """
        if category not in self.mapping:
            raise IndexError('Cannot find ' + category + ' in the vocab', category)
        return self.mapping[category]

    def has(self, category):
        """
        Returns true if the category is in the vocabulary
        """
        return category in self.mapping

    def update(self, categories):
        """
        Categories are added from an iterable or added-in from another mapping. Also, the iterable 
        is expected to be a sequence of elements, not a sequence of (key, value) pairs.
        """
        if isinstance(categories, (list, set, tuple)):
            for category in categories:
                self.add(category)
        elif isinstance(categories, dict):
            for category in categories:
                self.add(category, categories[category])
        else:
            raise TypeError()

    def add(self, category, count=1):
        """Adds count of the category to the frequency table.

        Args:
            category: string or integer, category to add frequency to.
            count(int): optional integer, how many to add.
        """
        if category not in self.mapping:
            if self.is_frozen:
                raise RuntimeError('Vocab is frozen')
            self.mapping[category] = len(self.mapping)
            self.reverse_mapping.append(category)
        self.freq[category] += count

    def reverse(self, index):
        """Given index, reverse to the category.

        Args:
          index(int): index of the category

        Returns:
            category

        Raises:
            ValueError: if this vocabulary wasn't initalized with support_reverse.
        """
        return self.reverse_mapping[index]

    @property
    def index2category(self):
        """
        Returns:
            dict: {idx, tok}
        """
        return {i: v for i, v in enumerate(self.reverse_mapping)}

    @property
    def category2index(self):
        """
        Returns:
            dict: {tok, idx}
        """
        return dict(self.mapping)
