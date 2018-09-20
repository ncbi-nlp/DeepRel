import logging

from nltk import sent_tokenize


def split_text(text):
    logger = logging.getLogger(__name__)

    sent_list = sent_tokenize(text)
    offset = 0
    for sent in sent_list:
        offset = text.find(sent, offset)
        if offset == -1:
            logger.debug('Cannot find {} in {}'.format(sent, text))
        yield sent, offset
        offset += len(sent)
