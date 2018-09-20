def Ptb2Dep(representation='CCprocessed', universal=False):
    """
    Convert ptb trees to universal dependencies
    """
    from .ptb2dep import Ptb2Dep
    return Ptb2Dep(representation=representation, universal=universal)


def GeniaTagger(genia_path):
    """
    Genia Tagger
    
    Args:
        genia_path: genia tagger path
    """
    from .geniatagger import GeniaTagger
    return GeniaTagger(genia_path)


def BllipParser(model_dir: str or None):
    """
    Bllip parser

    Args:
        model_dir: the location to download the model.
    """
    from .bllip_parser import Bllip
    return Bllip(model_dir=model_dir)