"""
Usage:
    cat_jsonl <input> <line>...

Options:
    --verbose
"""
import json

import docopt

if __name__ == '__main__':
    argv = docopt.docopt(__doc__)

    lines = [int(i) for i in argv['<line>']]
    with open(argv['<input>']) as fp:
        for i, line in enumerate(fp):
            if i in line:
                obj = json.loads(line)
                print(json.dumps(obj, indent=2))
