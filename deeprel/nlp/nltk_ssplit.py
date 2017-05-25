import logging

import bioc
from nltk import sent_tokenize


class NltkSSplitter(object):
    """
    NLTK sentence splitter
    """
    def split_c(self, collection):
        for document in collection.documents:
            self.split_c(document)

    def split_s(self, text):
        logger = logging.getLogger(__name__)

        sent_list = sent_tokenize(text)
        offset = 0
        for sent in sent_list:
            offset = text.find(sent, offset)
            if offset == -1:
                logger.debug('Cannot find {} in {}'.format(sent, text))
            yield sent, offset
            offset += len(sent)

    def split_d(self, document):
        for passage in document.passages:
            for text, offset in self.split_s(passage.text):
                sentence = bioc.BioCSentence()
                sentence.offset = offset + passage.offset
                sentence.text = text
                passage.add_sentence(sentence)
            passage.text = None

    def split_f(self, src, dst):
        """
        Split passages.
    
        Args:
            src(str): source file name in BioC format
            dst(str): target file name in BioC format
        """
        logger = logging.getLogger(__name__)
        logger.info('Process file: {}'.format(src))

        with bioc.iterparse(src) as parser:
            with bioc.iterwrite(dst, parser.get_collection_info()) as writer:
                for document in parser:
                    self.split_d(document)
                    writer.writedocument(document)


