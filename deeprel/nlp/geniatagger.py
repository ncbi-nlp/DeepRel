import subprocess
from typing import List, Tuple, Dict

from utils import to_path


class GeniaTagger(object):
    def __init__(self, genia_path):
        self._tagger = None
        self.genia_path = to_path(genia_path)
        if not self.genia_path.exists():
            raise ValueError('{} does not exist'.format(self.genia_path))

    def __parse(self, text):
        if self._tagger is None:
            # lazy load
            cmd = './{}'.format(self.genia_path)
            self._tagger = subprocess.Popen(cmd,
                                            cwd=str(self.genia_path.parent),
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            bufsize=1,
                                            universal_newlines=True)
        for line in text.split('\n'):
            line += '\n'
            self._tagger.stdin.write(line)
            while True:
                r = self._tagger.stdout.readline()[:-1]
                if not r:
                    break
                yield tuple(r.split('\t'))

    def parse(self, s: str, offset: int = 0) -> List[Dict]:
        """Parses the sentence text using genia tagger.

        Args:
            s: one sentence
            offset: the offset of the sentence in a document. Default is 0

        Returns:
            a list of tokens containing word, base, part-of-speech,
                chunk, named entity, start and end offset
        """
        toks = []
        char_index = 0
        for word, base, pos, chunk, ne in self.__parse(s):
            start = s.find(word, char_index)
            toks.append({
                "word": word,
                'base': base,
                'pos': pos,
                'chunk': chunk,
                'ne': ne,
                'start': offset + start,
                'end': offset + start + len(word)
            })
            if start != -1:
                char_index = start + len(word)
        return toks
