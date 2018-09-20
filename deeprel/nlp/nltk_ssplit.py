import logging
from typing import Generator, Tuple

from nltk import sent_tokenize


def split_text(text: str) -> Generator[Tuple[str, int], None, None]:
    """Split text into sentences with offset"""
    sent_list = sent_tokenize(text)
    offset = 0
    for sent in sent_list:
        offset = text.find(sent, offset)
        if offset == -1:
            logging.debug('Cannot find {} in {}'.format(sent, text))
        yield sent, offset
        offset += len(sent)
