import logging
import bioc

from deeprel import utils
from deeprel.nlp import Ptb2Dep


class DependencyAdder(object):
    def __init__(self):
        self.p2d = Ptb2Dep(universal=True)

    def add_dependency(self, obj):
        # create bioc sentence
        sentence = bioc.BioCSentence()
        sentence.offset = 0
        sentence.text = obj['text']
        annotation = bioc.BioCAnnotation()
        annotation.infons['parse tree'] = obj['parse tree']
        sentence.add_annotation(annotation)

        self.p2d.convert_s(sentence)

        m = {}
        for i, tok in enumerate(obj['toks']):
            tok['id'] = i
            # find bioc annotation
            found = False
            for ann in sentence.annotations:
                loc = ann.total_span
                if utils.intersect((tok['start'], tok['end']),
                                   (loc.offset, loc.offset + loc.length)):
                    if ann.id in m:
                        logging.debug('Duplicated id mapping: %s', ann.id)
                    m[ann.id] = i
                    if 'ROOT' in ann.infons:
                        tok['ROOT'] = True
                    found = True
                    break
            if not found:
                logging.debug('Cannot find %s in \n%s', tok, obj['id'])

        for rel in sentence.relations:
            node0 = rel.nodes[0]
            node1 = rel.nodes[1]
            if node0.refid in m and node1.refid in m:
                if node0.role == 'governor':
                    gov = m[node0.refid]
                    dep = m[node1.refid]
                else:
                    gov = m[node1.refid]
                    dep = m[node0.refid]
                if gov == dep:
                    logging.debug('Discard self loop')
                    continue
                tok = obj['toks'][dep]
                if 'governor' in tok:
                    if tok['governor'] == gov:
                        pass
                    if 'extra' in rel.infons:
                        pass
                    else:
                        logging.debug('%s: Two heads: %s', obj['id'], str(rel))
                else:
                    tok['governor'] = gov
                    tok['dependency'] = rel.infons['dependency']
            else:
                ann0 = None
                ann1 = None
                for annotation in sentence.annotations:
                    if annotation.id == node0.refid:
                        ann0 = annotation
                    if annotation.id == node1.refid:
                        ann1 = annotation
                logging.debug('Cannot find %s or %s in sentence: %s', node0, node1, obj['id'])
                logging.debug('%s', ann0)
                logging.debug('%s', ann1)
