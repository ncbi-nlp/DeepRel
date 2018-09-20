import tempfile

from bllipparser import ModelFetcher
from bllipparser import RerankingParser
import logging
import os


class Bllip:
    def __init__(self, model_dir=None):
        if model_dir is None:
            logging.debug("downloading GENIA+PubMed model if necessary ...")
            model_dir = ModelFetcher.download_and_install_model(
                'GENIA+PubMed', os.path.join(tempfile.gettempdir(), 'models'))
        self.model_dir = os.path.expanduser(model_dir)

        logging.debug('loading model %s ...', self.model_dir)
        self.rrp = RerankingParser.from_unified_model_dir(self.model_dir)

    def parse(self, s:str):
        """Parse the sentence text using Reranking parser.

        Args:
            s: one sentence

        Returns:
            parse tree, ScoredParse object in RerankingParser

        Raises:
            ValueError
        """
        if not s:
            raise ValueError('Cannot parse empty sentence: {}'.format(s))
        try:
            nbest = self.rrp.parse(str(s))
            return str(nbest[0].ptb_parse)
        except:
            raise ValueError('Cannot parse sentence: %s' % s)
