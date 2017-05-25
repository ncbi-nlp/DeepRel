def Ptb2Dep(representation='CCprocessed', universal=False):
    """
    Convert ptb trees to universal dependencies
    """
    from .ptb2dep import Ptb2Dep
    return Ptb2Dep(representation=representation, universal=universal)


def NltkSSplitter():
    """
    NLTK sentence splitter
    """
    from .nltk_ssplit import NltkSSplitter
    return NltkSSplitter()


def GeniaTagger(genia_path):
    """
    Genia Tagger
    
    Args:
        genia_path: genia tagger path
    """
    from .geniatagger import GeniaTagger
    return GeniaTagger(genia_path)


def Bllip(model_dir=None):
    """
    Bllip parser
    
    Args:
        model_dir(str): the location to download the model.
    """
    from .bllip import Bllip
    return Bllip(model_dir=model_dir)


def StanfordParser(corenlp_jars):
    """
    Stanford parser

    Args:
        corenlp_jars(str): the location to download the model.
    """
    from .stanford_parser import StanfordParser
    return StanfordParser(corenlp_jars=corenlp_jars)