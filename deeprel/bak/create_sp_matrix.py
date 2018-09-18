"""
Usage:
    create_sp_matrix.py [options] --vocab=<file> --all=<directory> --output=<file> INPUT_FILE

Options:
    --verbose
    --vocab=<file>
    --all=<directory>
    --output=<file>
"""

from cli_utils import parse_args
from deeprel.create_matrix import MatrixCreater

if __name__ == '__main__':
    argv = parse_args(__doc__)
    mc = MatrixCreater(argv['--vocab'], argv['--all'])
    mc.create_matrix(argv['INPUT_FILE'], toks_name='shortest path')
    mc.save(argv['--output'])
