import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple

class SwitchboardTurn(list):
    """
    A specialized list object used to encode switchboard utterances.
    The elements of the list are the words in the utterance; and two
    attributes, ``speaker`` and ``id``, are provided to retrieve the
    spearker identifier and utterance id.  Note that utterance ids
    are only unique within a given discourse.
    """

    def __init__(self, words, speaker, id):
        if False:
            i = 10
            return i + 15
        list.__init__(self, words)
        self.speaker = speaker
        self.id = int(id)

    def __repr__(self):
        if False:
            return 10
        if len(self) == 0:
            text = ''
        elif isinstance(self[0], tuple):
            text = ' '.join(('%s/%s' % w for w in self))
        else:
            text = ' '.join(self)
        return f'<{self.speaker}.{self.id}: {text!r}>'

class SwitchboardCorpusReader(CorpusReader):
    _FILES = ['tagged']

    def __init__(self, root, tagset=None):
        if False:
            return 10
        CorpusReader.__init__(self, root, self._FILES)
        self._tagset = tagset

    def words(self):
        if False:
            print('Hello World!')
        return StreamBackedCorpusView(self.abspath('tagged'), self._words_block_reader)

    def tagged_words(self, tagset=None):
        if False:
            i = 10
            return i + 15

        def tagged_words_block_reader(stream):
            if False:
                i = 10
                return i + 15
            return self._tagged_words_block_reader(stream, tagset)
        return StreamBackedCorpusView(self.abspath('tagged'), tagged_words_block_reader)

    def turns(self):
        if False:
            print('Hello World!')
        return StreamBackedCorpusView(self.abspath('tagged'), self._turns_block_reader)

    def tagged_turns(self, tagset=None):
        if False:
            i = 10
            return i + 15

        def tagged_turns_block_reader(stream):
            if False:
                print('Hello World!')
            return self._tagged_turns_block_reader(stream, tagset)
        return StreamBackedCorpusView(self.abspath('tagged'), tagged_turns_block_reader)

    def discourses(self):
        if False:
            i = 10
            return i + 15
        return StreamBackedCorpusView(self.abspath('tagged'), self._discourses_block_reader)

    def tagged_discourses(self, tagset=False):
        if False:
            for i in range(10):
                print('nop')

        def tagged_discourses_block_reader(stream):
            if False:
                return 10
            return self._tagged_discourses_block_reader(stream, tagset)
        return StreamBackedCorpusView(self.abspath('tagged'), tagged_discourses_block_reader)

    def _discourses_block_reader(self, stream):
        if False:
            return 10
        return [[self._parse_utterance(u, include_tag=False) for b in read_blankline_block(stream) for u in b.split('\n') if u.strip()]]

    def _tagged_discourses_block_reader(self, stream, tagset=None):
        if False:
            print('Hello World!')
        return [[self._parse_utterance(u, include_tag=True, tagset=tagset) for b in read_blankline_block(stream) for u in b.split('\n') if u.strip()]]

    def _turns_block_reader(self, stream):
        if False:
            while True:
                i = 10
        return self._discourses_block_reader(stream)[0]

    def _tagged_turns_block_reader(self, stream, tagset=None):
        if False:
            for i in range(10):
                print('nop')
        return self._tagged_discourses_block_reader(stream, tagset)[0]

    def _words_block_reader(self, stream):
        if False:
            print('Hello World!')
        return sum(self._discourses_block_reader(stream)[0], [])

    def _tagged_words_block_reader(self, stream, tagset=None):
        if False:
            return 10
        return sum(self._tagged_discourses_block_reader(stream, tagset)[0], [])
    _UTTERANCE_RE = re.compile('(\\w+)\\.(\\d+)\\:\\s*(.*)')
    _SEP = '/'

    def _parse_utterance(self, utterance, include_tag, tagset=None):
        if False:
            print('Hello World!')
        m = self._UTTERANCE_RE.match(utterance)
        if m is None:
            raise ValueError('Bad utterance %r' % utterance)
        (speaker, id, text) = m.groups()
        words = [str2tuple(s, self._SEP) for s in text.split()]
        if not include_tag:
            words = [w for (w, t) in words]
        elif tagset and tagset != self._tagset:
            words = [(w, map_tag(self._tagset, tagset, t)) for (w, t) in words]
        return SwitchboardTurn(words, speaker, id)