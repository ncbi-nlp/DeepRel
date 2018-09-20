"""
Convert ptb trees to universal dependencies
"""
import logging

import StanfordDependencies
import bioc

import utils


class Ptb2Dep(object):
    """
    Convert ptb trees to universal dependencies
    """

    basic = 'basic'
    collapsed = 'collapsed'
    CCprocessed = 'CCprocessed',
    collapsedTree = 'collapsedTree'

    def __init__(self, representation='CCprocessed', universal=False):
        """
        Args:
            representation(str): Currently supported representations are
                'basic', 'collapsed', 'CCprocessed', and 'collapsedTree'
            universal(bool): if True, use universal dependencies if they're available
        """
        try:
            import jpype
            __backend = 'jpype'
        except ImportError:
            __backend = 'subprocess'
        self.__sd = StanfordDependencies.get_instance(backend=__backend)
        self.representation = representation
        self.universal = universal

    def convert(self, parse_tree):
        """
        Convert ptb trees in a BioC sentence
        
        Args:
            parse_tree(str): parse tree in PTB format
            
        Examples:
            (ROOT (NP (JJ hello) (NN world) (. !)))
        """
        dependency_graph = self.__sd.convert_tree(parse_tree,
                                                  representation=self.representation,
                                                  universal=self.universal)
        return dependency_graph

    def convert_s(self, sentence):
        """
        Convert ptb trees in a BioC sentence
        """
        if len(sentence.annotations) <= 0:
            return
        try:
            dependency_graph = self.convert(sentence.annotations[0].infons['parse tree'])
            anns, rels = convert_dg(dependency_graph, sentence.text, sentence.offset)
            sentence.annotations = anns
            sentence.relations = rels
        except:
            logging.exception('Cannot convert %s', sentence)
            return

    def add_dependency(self, obj):
        # create bioc sentence
        sentence = bioc.BioCSentence()
        sentence.offset = 0
        sentence.text = obj['text']
        annotation = bioc.BioCAnnotation()
        annotation.infons['parse tree'] = obj['parse tree']
        sentence.add_annotation(annotation)

        self.convert_s(sentence)

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


def adapt_value(value):
    """
    Adapt string in PTB
    """
    value = value.replace("-LRB-", "(")
    value = value.replace("-RRB-", ")")
    value = value.replace("-LSB-", "[")
    value = value.replace("-RSB-", "]")
    value = value.replace("-LCB-", "{")
    value = value.replace("-RCB-", "}")
    value = value.replace("-lrb-", "(")
    value = value.replace("-rrb-", ")")
    value = value.replace("-lsb-", "[")
    value = value.replace("-rsb-", "]")
    value = value.replace("``", "\"")
    value = value.replace("''", "\"")
    value = value.replace("`", "'")
    return value


def convert_dg(dependency_graph, text, offset, ann_index=0, rel_index=0):
    """
    Convert dependency graph to annotations and relations
    """
    annotations = []
    relations = []
    annotation_id_map = {}
    start = 0
    for node in dependency_graph:
        if node.index in annotation_id_map:
            continue
        node_form = node.form
        index = text.find(node_form, start)
        if index == -1:
            node_form = adapt_value(node.form)
            index = text.find(node_form, start)
            if index == -1:
                logging.debug('Cannot convert parse tree to dependency graph at %d\n%d\n%s',
                              start, offset, str(dependency_graph))
                continue

        ann = bioc.BioCAnnotation()
        ann.id = 'T{}'.format(ann_index)
        ann.text = node_form
        ann.infons['tag'] = node.pos

        start = index

        ann.add_location(bioc.BioCLocation(start + offset, len(node_form)))
        annotations.append(ann)
        annotation_id_map[node.index] = ann_index
        ann_index += 1
        start += len(node_form)

    for node in dependency_graph:
        if node.head == 0:
            ann = annotations[annotation_id_map[node.index]]
            ann.infons['ROOT'] = True
            continue
        relation = bioc.BioCRelation()
        relation.id = 'R{}'.format(rel_index)
        relation.infons['dependency'] = node.deprel
        if node.extra:
            relation.infons['extra'] = node.extra
        if node.index in annotation_id_map and node.head in annotation_id_map:
            relation.add_node(bioc.BioCNode('T{}'.format(annotation_id_map[node.index]), 'dependant'))
            relation.add_node(bioc.BioCNode('T{}'.format(annotation_id_map[node.head]), 'governor'))
            relations.append(relation)
            rel_index += 1

    return annotations, relations
