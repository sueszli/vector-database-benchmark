"""
A reader for corpora whose documents are in MTE format.
"""
import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView

def xpath(root, path, ns):
    if False:
        for i in range(10):
            print('nop')
    return root.findall(path, ns)

class MTECorpusView(XMLCorpusView):
    """
    Class for lazy viewing the MTE Corpus.
    """

    def __init__(self, fileid, tagspec, elt_handler=None):
        if False:
            i = 10
            return i + 15
        XMLCorpusView.__init__(self, fileid, tagspec, elt_handler)

    def read_block(self, stream, tagspec=None, elt_handler=None):
        if False:
            i = 10
            return i + 15
        return list(filter(lambda x: x is not None, XMLCorpusView.read_block(self, stream, tagspec, elt_handler)))

class MTEFileReader:
    """
    Class for loading the content of the multext-east corpus. It
    parses the xml files and does some tag-filtering depending on the
    given method parameters.
    """
    ns = {'tei': 'https://www.tei-c.org/ns/1.0', 'xml': 'https://www.w3.org/XML/1998/namespace'}
    tag_ns = '{https://www.tei-c.org/ns/1.0}'
    xml_ns = '{https://www.w3.org/XML/1998/namespace}'
    word_path = 'TEI/text/body/div/div/p/s/(w|c)'
    sent_path = 'TEI/text/body/div/div/p/s'
    para_path = 'TEI/text/body/div/div/p'

    def __init__(self, file_path):
        if False:
            for i in range(10):
                print('nop')
        self.__file_path = file_path

    @classmethod
    def _word_elt(cls, elt, context):
        if False:
            for i in range(10):
                print('nop')
        return elt.text

    @classmethod
    def _sent_elt(cls, elt, context):
        if False:
            return 10
        return [cls._word_elt(w, None) for w in xpath(elt, '*', cls.ns)]

    @classmethod
    def _para_elt(cls, elt, context):
        if False:
            for i in range(10):
                print('nop')
        return [cls._sent_elt(s, None) for s in xpath(elt, '*', cls.ns)]

    @classmethod
    def _tagged_word_elt(cls, elt, context):
        if False:
            while True:
                i = 10
        if 'ana' not in elt.attrib:
            return (elt.text, '')
        if cls.__tags == '' and cls.__tagset == 'msd':
            return (elt.text, elt.attrib['ana'])
        elif cls.__tags == '' and cls.__tagset == 'universal':
            return (elt.text, MTETagConverter.msd_to_universal(elt.attrib['ana']))
        else:
            tags = re.compile('^' + re.sub('-', '.', cls.__tags) + '.*$')
            if tags.match(elt.attrib['ana']):
                if cls.__tagset == 'msd':
                    return (elt.text, elt.attrib['ana'])
                else:
                    return (elt.text, MTETagConverter.msd_to_universal(elt.attrib['ana']))
            else:
                return None

    @classmethod
    def _tagged_sent_elt(cls, elt, context):
        if False:
            print('Hello World!')
        return list(filter(lambda x: x is not None, [cls._tagged_word_elt(w, None) for w in xpath(elt, '*', cls.ns)]))

    @classmethod
    def _tagged_para_elt(cls, elt, context):
        if False:
            i = 10
            return i + 15
        return list(filter(lambda x: x is not None, [cls._tagged_sent_elt(s, None) for s in xpath(elt, '*', cls.ns)]))

    @classmethod
    def _lemma_word_elt(cls, elt, context):
        if False:
            return 10
        if 'lemma' not in elt.attrib:
            return (elt.text, '')
        else:
            return (elt.text, elt.attrib['lemma'])

    @classmethod
    def _lemma_sent_elt(cls, elt, context):
        if False:
            return 10
        return [cls._lemma_word_elt(w, None) for w in xpath(elt, '*', cls.ns)]

    @classmethod
    def _lemma_para_elt(cls, elt, context):
        if False:
            i = 10
            return i + 15
        return [cls._lemma_sent_elt(s, None) for s in xpath(elt, '*', cls.ns)]

    def words(self):
        if False:
            while True:
                i = 10
        return MTECorpusView(self.__file_path, MTEFileReader.word_path, MTEFileReader._word_elt)

    def sents(self):
        if False:
            return 10
        return MTECorpusView(self.__file_path, MTEFileReader.sent_path, MTEFileReader._sent_elt)

    def paras(self):
        if False:
            i = 10
            return i + 15
        return MTECorpusView(self.__file_path, MTEFileReader.para_path, MTEFileReader._para_elt)

    def lemma_words(self):
        if False:
            while True:
                i = 10
        return MTECorpusView(self.__file_path, MTEFileReader.word_path, MTEFileReader._lemma_word_elt)

    def tagged_words(self, tagset, tags):
        if False:
            print('Hello World!')
        MTEFileReader.__tagset = tagset
        MTEFileReader.__tags = tags
        return MTECorpusView(self.__file_path, MTEFileReader.word_path, MTEFileReader._tagged_word_elt)

    def lemma_sents(self):
        if False:
            return 10
        return MTECorpusView(self.__file_path, MTEFileReader.sent_path, MTEFileReader._lemma_sent_elt)

    def tagged_sents(self, tagset, tags):
        if False:
            while True:
                i = 10
        MTEFileReader.__tagset = tagset
        MTEFileReader.__tags = tags
        return MTECorpusView(self.__file_path, MTEFileReader.sent_path, MTEFileReader._tagged_sent_elt)

    def lemma_paras(self):
        if False:
            for i in range(10):
                print('nop')
        return MTECorpusView(self.__file_path, MTEFileReader.para_path, MTEFileReader._lemma_para_elt)

    def tagged_paras(self, tagset, tags):
        if False:
            while True:
                i = 10
        MTEFileReader.__tagset = tagset
        MTEFileReader.__tags = tags
        return MTECorpusView(self.__file_path, MTEFileReader.para_path, MTEFileReader._tagged_para_elt)

class MTETagConverter:
    """
    Class for converting msd tags to universal tags, more conversion
    options are currently not implemented.
    """
    mapping_msd_universal = {'A': 'ADJ', 'S': 'ADP', 'R': 'ADV', 'C': 'CONJ', 'D': 'DET', 'N': 'NOUN', 'M': 'NUM', 'Q': 'PRT', 'P': 'PRON', 'V': 'VERB', '.': '.', '-': 'X'}

    @staticmethod
    def msd_to_universal(tag):
        if False:
            print('Hello World!')
        '\n        This function converts the annotation from the Multex-East to the universal tagset\n        as described in Chapter 5 of the NLTK-Book\n\n        Unknown Tags will be mapped to X. Punctuation marks are not supported in MSD tags, so\n        '
        indicator = tag[0] if not tag[0] == '#' else tag[1]
        if not indicator in MTETagConverter.mapping_msd_universal:
            indicator = '-'
        return MTETagConverter.mapping_msd_universal[indicator]

class MTECorpusReader(TaggedCorpusReader):
    """
    Reader for corpora following the TEI-p5 xml scheme, such as MULTEXT-East.
    MULTEXT-East contains part-of-speech-tagged words with a quite precise tagging
    scheme. These tags can be converted to the Universal tagset
    """

    def __init__(self, root=None, fileids=None, encoding='utf8'):
        if False:
            i = 10
            return i + 15
        "\n        Construct a new MTECorpusreader for a set of documents\n        located at the given root directory.  Example usage:\n\n            >>> root = '/...path to corpus.../'\n            >>> reader = MTECorpusReader(root, 'oana-*.xml', 'utf8') # doctest: +SKIP\n\n        :param root: The root directory for this corpus. (default points to location in multext config file)\n        :param fileids: A list or regexp specifying the fileids in this corpus. (default is oana-en.xml)\n        :param encoding: The encoding of the given files (default is utf8)\n        "
        TaggedCorpusReader.__init__(self, root, fileids, encoding)
        self._readme = '00README.txt'

    def __fileids(self, fileids):
        if False:
            i = 10
            return i + 15
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        fileids = filter(lambda x: x in self._fileids, fileids)
        fileids = filter(lambda x: x not in ['oana-bg.xml', 'oana-mk.xml'], fileids)
        if not fileids:
            print('No valid multext-east file specified')
        return fileids

    def words(self, fileids=None):
        if False:
            i = 10
            return i + 15
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :return: the given file(s) as a list of words and punctuation symbols.\n        :rtype: list(str)\n        '
        return concat([MTEFileReader(os.path.join(self._root, f)).words() for f in self.__fileids(fileids)])

    def sents(self, fileids=None):
        if False:
            while True:
                i = 10
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :return: the given file(s) as a list of sentences or utterances,\n                 each encoded as a list of word strings\n        :rtype: list(list(str))\n        '
        return concat([MTEFileReader(os.path.join(self._root, f)).sents() for f in self.__fileids(fileids)])

    def paras(self, fileids=None):
        if False:
            print('Hello World!')
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :return: the given file(s) as a list of paragraphs, each encoded as a list\n                 of sentences, which are in turn encoded as lists of word string\n        :rtype: list(list(list(str)))\n        '
        return concat([MTEFileReader(os.path.join(self._root, f)).paras() for f in self.__fileids(fileids)])

    def lemma_words(self, fileids=None):
        if False:
            print('Hello World!')
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :return: the given file(s) as a list of words, the corresponding lemmas\n                 and punctuation symbols, encoded as tuples (word, lemma)\n        :rtype: list(tuple(str,str))\n        '
        return concat([MTEFileReader(os.path.join(self._root, f)).lemma_words() for f in self.__fileids(fileids)])

    def tagged_words(self, fileids=None, tagset='msd', tags=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :param tagset: The tagset that should be used in the returned object,\n                       either "universal" or "msd", "msd" is the default\n        :param tags: An MSD Tag that is used to filter all parts of the used corpus\n                     that are not more precise or at least equal to the given tag\n        :return: the given file(s) as a list of tagged words and punctuation symbols\n                 encoded as tuples (word, tag)\n        :rtype: list(tuple(str, str))\n        '
        if tagset == 'universal' or tagset == 'msd':
            return concat([MTEFileReader(os.path.join(self._root, f)).tagged_words(tagset, tags) for f in self.__fileids(fileids)])
        else:
            print('Unknown tagset specified.')

    def lemma_sents(self, fileids=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :return: the given file(s) as a list of sentences or utterances, each\n                 encoded as a list of tuples of the word and the corresponding\n                 lemma (word, lemma)\n        :rtype: list(list(tuple(str, str)))\n        '
        return concat([MTEFileReader(os.path.join(self._root, f)).lemma_sents() for f in self.__fileids(fileids)])

    def tagged_sents(self, fileids=None, tagset='msd', tags=''):
        if False:
            i = 10
            return i + 15
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :param tagset: The tagset that should be used in the returned object,\n                       either "universal" or "msd", "msd" is the default\n        :param tags: An MSD Tag that is used to filter all parts of the used corpus\n                     that are not more precise or at least equal to the given tag\n        :return: the given file(s) as a list of sentences or utterances, each\n                 each encoded as a list of (word,tag) tuples\n        :rtype: list(list(tuple(str, str)))\n        '
        if tagset == 'universal' or tagset == 'msd':
            return concat([MTEFileReader(os.path.join(self._root, f)).tagged_sents(tagset, tags) for f in self.__fileids(fileids)])
        else:
            print('Unknown tagset specified.')

    def lemma_paras(self, fileids=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :return: the given file(s) as a list of paragraphs, each encoded as a\n                 list of sentences, which are in turn encoded as a list of\n                 tuples of the word and the corresponding lemma (word, lemma)\n        :rtype: list(List(List(tuple(str, str))))\n        '
        return concat([MTEFileReader(os.path.join(self._root, f)).lemma_paras() for f in self.__fileids(fileids)])

    def tagged_paras(self, fileids=None, tagset='msd', tags=''):
        if False:
            i = 10
            return i + 15
        '\n        :param fileids: A list specifying the fileids that should be used.\n        :param tagset: The tagset that should be used in the returned object,\n                       either "universal" or "msd", "msd" is the default\n        :param tags: An MSD Tag that is used to filter all parts of the used corpus\n                     that are not more precise or at least equal to the given tag\n        :return: the given file(s) as a list of paragraphs, each encoded as a\n                 list of sentences, which are in turn encoded as a list\n                 of (word,tag) tuples\n        :rtype: list(list(list(tuple(str, str))))\n        '
        if tagset == 'universal' or tagset == 'msd':
            return concat([MTEFileReader(os.path.join(self._root, f)).tagged_paras(tagset, tags) for f in self.__fileids(fileids)])
        else:
            print('Unknown tagset specified.')