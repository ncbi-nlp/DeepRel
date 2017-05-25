import logging
import tempfile

import bioc
import os
from bllipparser import ModelFetcher
from bllipparser import RerankingParser


class Bllip:
    def __init__(self, model_dir=None):
        logger = logging.getLogger(__name__)
        if not model_dir:
            logger.info("downloading GENIA+PubMed model if necessary ...")
            model_dir = ModelFetcher.download_and_install_model(
                'GENIA+PubMed', os.path.join(tempfile.gettempdir(), 'models'))
        self.model_dir = model_dir

        logging.debug('loading model %s ...' % model_dir)
        self.rrp = RerankingParser.from_unified_model_dir(model_dir)

    def parse(self, s):
        """Parse the sentence text using Reranking parser.
        
        Args:
            s(str): one sentence
            
        Returns:
            ScoredParse: parse tree, ScoredParse object in RerankingParser; None if failed
        """
        logger = logging.getLogger(__name__)
        logger.debug('parse: %s' % s)
        try:
            nbest = self.rrp.parse(s)
            return nbest[0].ptb_parse
        except:
            logger.exception('Cannot parse sentence: {}'.format(s))
            return None

    def parsef(self, src, dst, sentence_filter=lambda x: True):
        """
        Parse sentences in BioC format when sentence_filter returns true
        
        Args:
            src(str): source file name in BioC format
            dst(str): target file name in BioC format
            sentence_filter: only parse the sentence when sentence_filter returns true
        """
        logger = logging.getLogger(__name__)

        with bioc.iterparse(src) as parser:
            collection = parser.get_collection_info()
            collection.infons['tool'] = 'Bllip'
            collection.infons['process'] = 'parse'
            with bioc.iterwrite(dst, collection) as writer:
                id = 0
                for document in parser:
                    logger.debug('Parse document: %s' % document.id)
                    for passage in document.passages:
                        for sentence in filter(sentence_filter, passage.sentences):
                            try:
                                text = sentence.text
                                if not text:
                                    continue
                                tree = self.parse(text)
                                if not tree:
                                    continue
                                annotation = bioc.BioCAnnotation()
                                annotation.id = 'T{}'.format(id)
                                annotation.text = sentence.text
                                annotation.infons['parse tree'] = tree
                                annotation.add_location(
                                    bioc.BioCLocation(sentence.offset, len(sentence.text)))
                                sentence.add_annotation(annotation)
                                id += 1
                            except:
                                logger.exception(
                                    'Some errors for parsing sentence: {}'.format(sentence.offset))
                    writer.writedocument(document)
