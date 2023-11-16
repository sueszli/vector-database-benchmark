import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree

class PropbankCorpusReader(CorpusReader):
    """
    Corpus reader for the propbank corpus, which augments the Penn
    Treebank with information about the predicate argument structure
    of every verb instance.  The corpus consists of two parts: the
    predicate-argument annotations themselves, and a set of "frameset
    files" which define the argument labels used by the annotations,
    on a per-verb basis.  Each "frameset file" contains one or more
    predicates, such as ``'turn'`` or ``'turn_on'``, each of which is
    divided into coarse-grained word senses called "rolesets".  For
    each "roleset", the frameset file provides descriptions of the
    argument roles, along with examples.
    """

    def __init__(self, root, propfile, framefiles='', verbsfile=None, parse_fileid_xform=None, parse_corpus=None, encoding='utf8'):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param root: The root directory for this corpus.\n        :param propfile: The name of the file containing the predicate-\n            argument annotations (relative to ``root``).\n        :param framefiles: A list or regexp specifying the frameset\n            fileids for this corpus.\n        :param parse_fileid_xform: A transform that should be applied\n            to the fileids in this corpus.  This should be a function\n            of one argument (a fileid) that returns a string (the new\n            fileid).\n        :param parse_corpus: The corpus containing the parse trees\n            corresponding to this corpus.  These parse trees are\n            necessary to resolve the tree pointers used by propbank.\n        '
        if isinstance(framefiles, str):
            framefiles = find_corpus_fileids(root, framefiles)
        framefiles = list(framefiles)
        CorpusReader.__init__(self, root, [propfile, verbsfile] + framefiles, encoding)
        self._propfile = propfile
        self._framefiles = framefiles
        self._verbsfile = verbsfile
        self._parse_fileid_xform = parse_fileid_xform
        self._parse_corpus = parse_corpus

    def instances(self, baseform=None):
        if False:
            print('Hello World!')
        '\n        :return: a corpus view that acts as a list of\n            ``PropBankInstance`` objects, one for each noun in the corpus.\n        '
        kwargs = {}
        if baseform is not None:
            kwargs['instance_filter'] = lambda inst: inst.baseform == baseform
        return StreamBackedCorpusView(self.abspath(self._propfile), lambda stream: self._read_instance_block(stream, **kwargs), encoding=self.encoding(self._propfile))

    def lines(self):
        if False:
            print('Hello World!')
        '\n        :return: a corpus view that acts as a list of strings, one for\n            each line in the predicate-argument annotation file.\n        '
        return StreamBackedCorpusView(self.abspath(self._propfile), read_line_block, encoding=self.encoding(self._propfile))

    def roleset(self, roleset_id):
        if False:
            print('Hello World!')
        '\n        :return: the xml description for the given roleset.\n        '
        baseform = roleset_id.split('.')[0]
        framefile = 'frames/%s.xml' % baseform
        if framefile not in self._framefiles:
            raise ValueError('Frameset file for %s not found' % roleset_id)
        with self.abspath(framefile).open() as fp:
            etree = ElementTree.parse(fp).getroot()
        for roleset in etree.findall('predicate/roleset'):
            if roleset.attrib['id'] == roleset_id:
                return roleset
        raise ValueError(f'Roleset {roleset_id} not found in {framefile}')

    def rolesets(self, baseform=None):
        if False:
            while True:
                i = 10
        '\n        :return: list of xml descriptions for rolesets.\n        '
        if baseform is not None:
            framefile = 'frames/%s.xml' % baseform
            if framefile not in self._framefiles:
                raise ValueError('Frameset file for %s not found' % baseform)
            framefiles = [framefile]
        else:
            framefiles = self._framefiles
        rsets = []
        for framefile in framefiles:
            with self.abspath(framefile).open() as fp:
                etree = ElementTree.parse(fp).getroot()
            rsets.append(etree.findall('predicate/roleset'))
        return LazyConcatenation(rsets)

    def verbs(self):
        if False:
            i = 10
            return i + 15
        '\n        :return: a corpus view that acts as a list of all verb lemmas\n            in this corpus (from the verbs.txt file).\n        '
        return StreamBackedCorpusView(self.abspath(self._verbsfile), read_line_block, encoding=self.encoding(self._verbsfile))

    def _read_instance_block(self, stream, instance_filter=lambda inst: True):
        if False:
            return 10
        block = []
        for i in range(100):
            line = stream.readline().strip()
            if line:
                inst = PropbankInstance.parse(line, self._parse_fileid_xform, self._parse_corpus)
                if instance_filter(inst):
                    block.append(inst)
        return block

class PropbankInstance:

    def __init__(self, fileid, sentnum, wordnum, tagger, roleset, inflection, predicate, arguments, parse_corpus=None):
        if False:
            for i in range(10):
                print('nop')
        self.fileid = fileid
        "The name of the file containing the parse tree for this\n        instance's sentence."
        self.sentnum = sentnum
        'The sentence number of this sentence within ``fileid``.\n        Indexing starts from zero.'
        self.wordnum = wordnum
        "The word number of this instance's predicate within its\n        containing sentence.  Word numbers are indexed starting from\n        zero, and include traces and other empty parse elements."
        self.tagger = tagger
        "An identifier for the tagger who tagged this instance; or\n        ``'gold'`` if this is an adjuticated instance."
        self.roleset = roleset
        "The name of the roleset used by this instance's predicate.\n        Use ``propbank.roleset() <PropbankCorpusReader.roleset>`` to\n        look up information about the roleset."
        self.inflection = inflection
        "A ``PropbankInflection`` object describing the inflection of\n        this instance's predicate."
        self.predicate = predicate
        "A ``PropbankTreePointer`` indicating the position of this\n        instance's predicate within its containing sentence."
        self.arguments = tuple(arguments)
        "A list of tuples (argloc, argid), specifying the location\n        and identifier for each of the predicate's argument in the\n        containing sentence.  Argument identifiers are strings such as\n        ``'ARG0'`` or ``'ARGM-TMP'``.  This list does *not* contain\n        the predicate."
        self.parse_corpus = parse_corpus
        'A corpus reader for the parse trees corresponding to the\n        instances in this propbank corpus.'

    @property
    def baseform(self):
        if False:
            while True:
                i = 10
        'The baseform of the predicate.'
        return self.roleset.split('.')[0]

    @property
    def sensenumber(self):
        if False:
            print('Hello World!')
        'The sense number of the predicate.'
        return self.roleset.split('.')[1]

    @property
    def predid(self):
        if False:
            print('Hello World!')
        'Identifier of the predicate.'
        return 'rel'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<PropbankInstance: {}, sent {}, word {}>'.format(self.fileid, self.sentnum, self.wordnum)

    def __str__(self):
        if False:
            return 10
        s = '{} {} {} {} {} {}'.format(self.fileid, self.sentnum, self.wordnum, self.tagger, self.roleset, self.inflection)
        items = self.arguments + ((self.predicate, 'rel'),)
        for (argloc, argid) in sorted(items):
            s += f' {argloc}-{argid}'
        return s

    def _get_tree(self):
        if False:
            return 10
        if self.parse_corpus is None:
            return None
        if self.fileid not in self.parse_corpus.fileids():
            return None
        return self.parse_corpus.parsed_sents(self.fileid)[self.sentnum]
    tree = property(_get_tree, doc='\n        The parse tree corresponding to this instance, or None if\n        the corresponding tree is not available.')

    @staticmethod
    def parse(s, parse_fileid_xform=None, parse_corpus=None):
        if False:
            i = 10
            return i + 15
        pieces = s.split()
        if len(pieces) < 7:
            raise ValueError('Badly formatted propbank line: %r' % s)
        (fileid, sentnum, wordnum, tagger, roleset, inflection) = pieces[:6]
        rel = [p for p in pieces[6:] if p.endswith('-rel')]
        args = [p for p in pieces[6:] if not p.endswith('-rel')]
        if len(rel) != 1:
            raise ValueError('Badly formatted propbank line: %r' % s)
        if parse_fileid_xform is not None:
            fileid = parse_fileid_xform(fileid)
        sentnum = int(sentnum)
        wordnum = int(wordnum)
        inflection = PropbankInflection.parse(inflection)
        predicate = PropbankTreePointer.parse(rel[0][:-4])
        arguments = []
        for arg in args:
            (argloc, argid) = arg.split('-', 1)
            arguments.append((PropbankTreePointer.parse(argloc), argid))
        return PropbankInstance(fileid, sentnum, wordnum, tagger, roleset, inflection, predicate, arguments, parse_corpus)

class PropbankPointer:
    """
    A pointer used by propbank to identify one or more constituents in
    a parse tree.  ``PropbankPointer`` is an abstract base class with
    three concrete subclasses:

      - ``PropbankTreePointer`` is used to point to single constituents.
      - ``PropbankSplitTreePointer`` is used to point to 'split'
        constituents, which consist of a sequence of two or more
        ``PropbankTreePointer`` pointers.
      - ``PropbankChainTreePointer`` is used to point to entire trace
        chains in a tree.  It consists of a sequence of pieces, which
        can be ``PropbankTreePointer`` or ``PropbankSplitTreePointer`` pointers.
    """

    def __init__(self):
        if False:
            return 10
        if self.__class__ == PropbankPointer:
            raise NotImplementedError()

class PropbankChainTreePointer(PropbankPointer):

    def __init__(self, pieces):
        if False:
            return 10
        self.pieces = pieces
        'A list of the pieces that make up this chain.  Elements may\n           be either ``PropbankSplitTreePointer`` or\n           ``PropbankTreePointer`` pointers.'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '*'.join(('%s' % p for p in self.pieces))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<PropbankChainTreePointer: %s>' % self

    def select(self, tree):
        if False:
            return 10
        if tree is None:
            raise ValueError('Parse tree not available')
        return Tree('*CHAIN*', [p.select(tree) for p in self.pieces])

class PropbankSplitTreePointer(PropbankPointer):

    def __init__(self, pieces):
        if False:
            i = 10
            return i + 15
        self.pieces = pieces
        'A list of the pieces that make up this chain.  Elements are\n           all ``PropbankTreePointer`` pointers.'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return ','.join(('%s' % p for p in self.pieces))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<PropbankSplitTreePointer: %s>' % self

    def select(self, tree):
        if False:
            for i in range(10):
                print('nop')
        if tree is None:
            raise ValueError('Parse tree not available')
        return Tree('*SPLIT*', [p.select(tree) for p in self.pieces])

@total_ordering
class PropbankTreePointer(PropbankPointer):
    """
    wordnum:height*wordnum:height*...
    wordnum:height,

    """

    def __init__(self, wordnum, height):
        if False:
            i = 10
            return i + 15
        self.wordnum = wordnum
        self.height = height

    @staticmethod
    def parse(s):
        if False:
            for i in range(10):
                print('nop')
        pieces = s.split('*')
        if len(pieces) > 1:
            return PropbankChainTreePointer([PropbankTreePointer.parse(elt) for elt in pieces])
        pieces = s.split(',')
        if len(pieces) > 1:
            return PropbankSplitTreePointer([PropbankTreePointer.parse(elt) for elt in pieces])
        pieces = s.split(':')
        if len(pieces) != 2:
            raise ValueError('bad propbank pointer %r' % s)
        return PropbankTreePointer(int(pieces[0]), int(pieces[1]))

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.wordnum}:{self.height}'

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'PropbankTreePointer(%d, %d)' % (self.wordnum, self.height)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        while isinstance(other, (PropbankChainTreePointer, PropbankSplitTreePointer)):
            other = other.pieces[0]
        if not isinstance(other, PropbankTreePointer):
            return self is other
        return self.wordnum == other.wordnum and self.height == other.height

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def __lt__(self, other):
        if False:
            print('Hello World!')
        while isinstance(other, (PropbankChainTreePointer, PropbankSplitTreePointer)):
            other = other.pieces[0]
        if not isinstance(other, PropbankTreePointer):
            return id(self) < id(other)
        return (self.wordnum, -self.height) < (other.wordnum, -other.height)

    def select(self, tree):
        if False:
            print('Hello World!')
        if tree is None:
            raise ValueError('Parse tree not available')
        return tree[self.treepos(tree)]

    def treepos(self, tree):
        if False:
            print('Hello World!')
        "\n        Convert this pointer to a standard 'tree position' pointer,\n        given that it points to the given tree.\n        "
        if tree is None:
            raise ValueError('Parse tree not available')
        stack = [tree]
        treepos = []
        wordnum = 0
        while True:
            if isinstance(stack[-1], Tree):
                if len(treepos) < len(stack):
                    treepos.append(0)
                else:
                    treepos[-1] += 1
                if treepos[-1] < len(stack[-1]):
                    stack.append(stack[-1][treepos[-1]])
                else:
                    stack.pop()
                    treepos.pop()
            elif wordnum == self.wordnum:
                return tuple(treepos[:len(treepos) - self.height - 1])
            else:
                wordnum += 1
                stack.pop()

class PropbankInflection:
    INFINITIVE = 'i'
    GERUND = 'g'
    PARTICIPLE = 'p'
    FINITE = 'v'
    FUTURE = 'f'
    PAST = 'p'
    PRESENT = 'n'
    PERFECT = 'p'
    PROGRESSIVE = 'o'
    PERFECT_AND_PROGRESSIVE = 'b'
    THIRD_PERSON = '3'
    ACTIVE = 'a'
    PASSIVE = 'p'
    NONE = '-'

    def __init__(self, form='-', tense='-', aspect='-', person='-', voice='-'):
        if False:
            i = 10
            return i + 15
        self.form = form
        self.tense = tense
        self.aspect = aspect
        self.person = person
        self.voice = voice

    def __str__(self):
        if False:
            return 10
        return self.form + self.tense + self.aspect + self.person + self.voice

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<PropbankInflection: %s>' % self
    _VALIDATE = re.compile('[igpv\\-][fpn\\-][pob\\-][3\\-][ap\\-]$')

    @staticmethod
    def parse(s):
        if False:
            print('Hello World!')
        if not isinstance(s, str):
            raise TypeError('expected a string')
        if len(s) != 5 or not PropbankInflection._VALIDATE.match(s):
            raise ValueError('Bad propbank inflection string %r' % s)
        return PropbankInflection(*s)