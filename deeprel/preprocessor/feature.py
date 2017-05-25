import copy
import sys
import logging
import collections
import networkx as nx

import numpy as np


def generate(obj):
    relations = obj['relations']
    protein_mentions = obj['annotations']
    retoken_results = obj['toks']

    examples = []
    for idx, relation in enumerate(relations):

        arg1id = relation['arg1']
        arg2id = relation['arg2']

        toks = copy.deepcopy(retoken_results)

        arg1_indices = arg_indices(arg1id, protein_mentions, toks)
        arg2_indices = arg_indices(arg2id, protein_mentions, toks)

        others_indices = other_indices(protein_mentions, toks)

        for i, tok in enumerate(toks):
            # distance feature
            arg1_dis = get_min_dis(i, arg1_indices)
            arg2_dis = get_min_dis(i, arg2_indices)
            # type feature
            t = 'O'
            if i in arg1_indices:
                t = 'Arg_1'
            elif i in arg2_indices:
                t = 'Arg_2'
            elif i in others_indices:
                t = 'Arg_O'
            tok['type'] = t
            tok['arg1_dis'] = arg1_dis
            tok['arg2_dis'] = arg2_dis
        ex = collections.OrderedDict({
            'id': relation['id'],
            'arg1': arg1id,
            'arg2': arg2id,
            'label': relation['label'],
            'toks': toks,
        })

        # shortest path
        toks = copy.deepcopy(retoken_results)
        if arg1_indices and arg2_indices:
            toks = get_shortest_path(toks, arg1_indices[-1], arg2_indices[-1])
            arg1_indices = arg_indices(arg1id, protein_mentions, toks)
            arg2_indices = arg_indices(arg2id, protein_mentions, toks)

        others_indices = other_indices(protein_mentions, toks)

        for i, tok in enumerate(toks):
            # distance feature
            arg1_dis = get_min_dis(i, arg1_indices)
            arg2_dis = get_min_dis(i, arg2_indices)
            # type feature
            t = 'O'
            if i in arg1_indices:
                t = 'Arg_1'
            elif i in arg2_indices:
                t = 'Arg_2'
            elif i in others_indices:
                t = 'Arg_O'
            tok['type'] = t
            tok['arg1_dis'] = arg1_dis
            tok['arg2_dis'] = arg2_dis
        ex['shortest path'] = toks
        examples.append(ex)

    return examples


def get_shortest_path(toks, source, target):
    # construct graph
    graph = load(toks)

    if toks[source]['start'] > toks[target]['start']:
        tmp = source
        source = target
        target = tmp

    try:
        path = nx.shortest_path(graph, toks[source]['id'], toks[target]['id'])
        subtoks = []
        for i in sorted(path):
            subtoks.append(toks[i])

        # re index
        d = {tok['id']: i for i, tok in enumerate(subtoks)}
        for tok in subtoks:
            tok['id'] = d[tok['id']]
            if 'governor' in tok and tok['governor'] not in path:
                del tok['governor']
            else:
                tok['governor'] = d[tok['governor']]

        logging.debug("Successful find a shortest path: %s", path)
        return subtoks
    except:
        return toks


def load(toks):
    graph = nx.Graph()

    for tok in toks:
        graph.add_node(tok['id'])

    for tok in toks:
        if 'governor' in tok:
            governor = tok['governor']
            dependant = tok['id']
            graph.add_edge(governor, dependant)
    return graph


def get_min_dis(i, indices):
    if not indices:
        return -sys.maxsize
    distances = np.array([abs(i-v) for v in indices])
    min_idx = np.argmin(distances)
    return i - indices[min_idx]


def other_indices(mentions, toks):
    """
    Get indices of toks where toks[i].start == mention.start
    """
    indices = []
    for i, tok in enumerate(toks):
        for m in mentions:
            if m['start'] <= tok['start'] and tok['end'] <= m['end']:
                indices.append(i)
                break
    return indices


def arg_indices(concept, mentions, toks):
    """
    Get indices of toks where toks[i].start == mention.start
    """
    indices = []
    for i, tok in enumerate(toks):
        for m in mentions:
            if 'start' not in m:
                logging.warning('%s', m)
            if m['id'] == concept and m['start'] <= tok['start'] and tok['end'] <= m['end']:
                indices.append(i)
                break
    return indices
