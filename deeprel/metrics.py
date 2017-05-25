from __future__ import print_function, division

import copy
import numpy as np

PLAIN = 'plain'
MARKDOWN = 'markdown'
TAB = 'tab'
COMMA = 'comma'


def classification_report(total, digits=4, sort_keys=False, type=PLAIN):
    """
    Get a text report showing the main classification metrics

    Args:
        total(dict): {label: [tp, fp, fn]}
        digits(int): digit accuracy, default 2
        sort_keys(bool): if true, sort the rows by labels
        type(str): one of plain, markdown, tab, or comma

    Returns:
        str: the report
    """
    total = [[k, total[k][0], total[k][1], total[k][2]] for k in total]

    separator = '  '
    adjust = True
    min_width = 0
    if type == PLAIN:
        pass
    elif type == MARKDOWN:
        separator = '|'
        min_width = 4
    elif type == TAB:
        separator = '\t'
        adjust = False
    elif type == COMMA:
        separator = ','
        adjust = False
    else:
        raise ValueError('type should be one of plain, markdown, tab, or comma')

    total = copy.deepcopy(total)
    if sort_keys:
        sorted(total, key=lambda row: row[0])

    # calculate avg / total
    total.append(['Total',
                  np.sum([row[1] for row in total]),
                  np.sum([row[2] for row in total]),
                  np.sum([row[3] for row in total])])

    # calculate p, r and f
    for row in total:
        p = row[1] / (row[1] + row[2]) if row[1] + row[2] != 0 else 0
        r = row[1] / (row[1] + row[3]) if row[1] + row[3] != 0 else 0
        f = 2 * p * r / (p + r) if p + r != 0 else 0
        row += [p, r, f]

    # to string
    total_str = [['Label', 'TP', 'FP', 'FN', "precision", "recall", "f1-score"]]
    for row in total:
        row_str = [row[0]]
        for v in row[1:4]:
            row_str.append(str(v))
        for v in row[4:]:
            row_str.append('{0:0.{1}f}'.format(v, digits))
        total_str.append(row_str)

    if adjust:
        # calculate width
        widths = []
        for i in range(len(total_str[0])):
            widths.append(max(min_width, max(len(row[i]) for row in total_str)))

        # adjust width
        for row in total_str:
            for i, v in enumerate(row):
                if i == 0:
                    row[i] = v.ljust(widths[i])
                else:
                    row[i] = v.rjust(widths[i])

    report = ''
    for i, row in enumerate(total_str):
        report += separator.join(row) + '\n'
        if type == 'markdown' and i == 0:
            report += len(row[0]) * '-' + separator
            report += separator.join([(len(v)-1)*'-' + ':' for v in row[1:]]) + '\n'
    return report
