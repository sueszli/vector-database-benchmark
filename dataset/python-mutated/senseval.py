"""
Read from the Senseval 2 Corpus.

SENSEVAL [http://www.senseval.org/]
Evaluation exercises for Word Sense Disambiguation.
Organized by ACL-SIGLEX [https://www.siglex.org/]

Prepared by Ted Pedersen <tpederse@umn.edu>, University of Minnesota,
https://www.d.umn.edu/~tpederse/data.html
Distributed with permission.

The NLTK version of the Senseval 2 files uses well-formed XML.
Each instance of the ambiguous words "hard", "interest", "line", and "serve"
is tagged with a sense identifier, and supplied with context.
"""
import re
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *

class SensevalInstance:

    def __init__(self, word, position, context, senses):
        if False:
            for i in range(10):
                print('nop')
        self.word = word
        self.senses = tuple(senses)
        self.position = position
        self.context = context

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'SensevalInstance(word=%r, position=%r, context=%r, senses=%r)' % (self.word, self.position, self.context, self.senses)

class SensevalCorpusReader(CorpusReader):

    def instances(self, fileids=None):
        if False:
            print('Hello World!')
        return concat([SensevalCorpusView(fileid, enc) for (fileid, enc) in self.abspaths(fileids, True)])

    def _entry(self, tree):
        if False:
            for i in range(10):
                print('nop')
        elts = []
        for lexelt in tree.findall('lexelt'):
            for inst in lexelt.findall('instance'):
                sense = inst[0].attrib['senseid']
                context = [(w.text, w.attrib['pos']) for w in inst[1]]
                elts.append((sense, context))
        return elts

class SensevalCorpusView(StreamBackedCorpusView):

    def __init__(self, fileid, encoding):
        if False:
            i = 10
            return i + 15
        StreamBackedCorpusView.__init__(self, fileid, encoding=encoding)
        self._word_tokenizer = WhitespaceTokenizer()
        self._lexelt_starts = [0]
        self._lexelts = [None]

    def read_block(self, stream):
        if False:
            while True:
                i = 10
        lexelt_num = bisect.bisect_right(self._lexelt_starts, stream.tell()) - 1
        lexelt = self._lexelts[lexelt_num]
        instance_lines = []
        in_instance = False
        while True:
            line = stream.readline()
            if line == '':
                assert instance_lines == []
                return []
            if line.lstrip().startswith('<lexelt'):
                lexelt_num += 1
                m = re.search('item=("[^"]+"|\'[^\']+\')', line)
                assert m is not None
                lexelt = m.group(1)[1:-1]
                if lexelt_num < len(self._lexelts):
                    assert lexelt == self._lexelts[lexelt_num]
                else:
                    self._lexelts.append(lexelt)
                    self._lexelt_starts.append(stream.tell())
            if line.lstrip().startswith('<instance'):
                assert instance_lines == []
                in_instance = True
            if in_instance:
                instance_lines.append(line)
            if line.lstrip().startswith('</instance'):
                xml_block = '\n'.join(instance_lines)
                xml_block = _fixXML(xml_block)
                inst = ElementTree.fromstring(xml_block)
                return [self._parse_instance(inst, lexelt)]

    def _parse_instance(self, instance, lexelt):
        if False:
            while True:
                i = 10
        senses = []
        context = []
        position = None
        for child in instance:
            if child.tag == 'answer':
                senses.append(child.attrib['senseid'])
            elif child.tag == 'context':
                context += self._word_tokenizer.tokenize(child.text)
                for cword in child:
                    if cword.tag == 'compound':
                        cword = cword[0]
                    if cword.tag == 'head':
                        assert position is None, 'head specified twice'
                        assert cword.text.strip() or len(cword) == 1
                        assert not (cword.text.strip() and len(cword) == 1)
                        position = len(context)
                        if cword.text.strip():
                            context.append(cword.text.strip())
                        elif cword[0].tag == 'wf':
                            context.append((cword[0].text, cword[0].attrib['pos']))
                            if cword[0].tail:
                                context += self._word_tokenizer.tokenize(cword[0].tail)
                        else:
                            assert False, 'expected CDATA or wf in <head>'
                    elif cword.tag == 'wf':
                        context.append((cword.text, cword.attrib['pos']))
                    elif cword.tag == 's':
                        pass
                    else:
                        print('ACK', cword.tag)
                        assert False, 'expected CDATA or <wf> or <head>'
                    if cword.tail:
                        context += self._word_tokenizer.tokenize(cword.tail)
            else:
                assert False, 'unexpected tag %s' % child.tag
        return SensevalInstance(lexelt, position, context, senses)

def _fixXML(text):
    if False:
        while True:
            i = 10
    '\n    Fix the various issues with Senseval pseudo-XML.\n    '
    text = re.sub('<([~\\^])>', '\\1', text)
    text = re.sub('(\\s+)\\&(\\s+)', '\\1&amp;\\2', text)
    text = re.sub('"""', '\'"\'', text)
    text = re.sub('(<[^<]*snum=)([^">]+)>', '\\1"\\2"/>', text)
    text = re.sub('<\\&frasl>\\s*<p[^>]*>', 'FRASL', text)
    text = re.sub('<\\&I[^>]*>', '', text)
    text = re.sub('<{([^}]+)}>', '\\1', text)
    text = re.sub('<(@|/?p)>', '', text)
    text = re.sub('<&\\w+ \\.>', '', text)
    text = re.sub('<!DOCTYPE[^>]*>', '', text)
    text = re.sub('<\\[\\/?[^>]+\\]*>', '', text)
    text = re.sub('<(\\&\\w+;)>', '\\1', text)
    text = re.sub('&(?!amp|gt|lt|apos|quot)', '', text)
    text = re.sub('[ \\t]*([^<>\\s]+?)[ \\t]*<p="([^"]*"?)"/>', ' <wf pos="\\2">\\1</wf>', text)
    text = re.sub('\\s*"\\s*<p=\\\'"\\\'/>', ' <wf pos=\'"\'>"</wf>', text)
    return text