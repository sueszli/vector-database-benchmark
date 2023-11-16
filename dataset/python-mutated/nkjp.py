import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView

def _parse_args(fun):
    if False:
        return 10
    '\n    Wraps function arguments:\n    if fileids not specified then function set NKJPCorpusReader paths.\n    '

    @functools.wraps(fun)
    def decorator(self, fileids=None, **kwargs):
        if False:
            return 10
        if not fileids:
            fileids = self._paths
        return fun(self, fileids, **kwargs)
    return decorator

class NKJPCorpusReader(XMLCorpusReader):
    WORDS_MODE = 0
    SENTS_MODE = 1
    HEADER_MODE = 2
    RAW_MODE = 3

    def __init__(self, root, fileids='.*'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Corpus reader designed to work with National Corpus of Polish.\n        See http://nkjp.pl/ for more details about NKJP.\n        use example:\n        import nltk\n        import nkjp\n        from nkjp import NKJPCorpusReader\n        x = NKJPCorpusReader(root='/home/USER/nltk_data/corpora/nkjp/', fileids='') # obtain the whole corpus\n        x.header()\n        x.raw()\n        x.words()\n        x.tagged_words(tags=['subst', 'comp'])  #Link to find more tags: nkjp.pl/poliqarp/help/ense2.html\n        x.sents()\n        x = NKJPCorpusReader(root='/home/USER/nltk_data/corpora/nkjp/', fileids='Wilk*') # obtain particular file(s)\n        x.header(fileids=['WilkDom', '/home/USER/nltk_data/corpora/nkjp/WilkWilczy'])\n        x.tagged_words(fileids=['WilkDom', '/home/USER/nltk_data/corpora/nkjp/WilkWilczy'], tags=['subst', 'comp'])\n        "
        if isinstance(fileids, str):
            XMLCorpusReader.__init__(self, root, fileids + '.*/header.xml')
        else:
            XMLCorpusReader.__init__(self, root, [fileid + '/header.xml' for fileid in fileids])
        self._paths = self.get_paths()

    def get_paths(self):
        if False:
            for i in range(10):
                print('nop')
        return [os.path.join(str(self._root), f.split('header.xml')[0]) for f in self._fileids]

    def fileids(self):
        if False:
            return 10
        '\n        Returns a list of file identifiers for the fileids that make up\n        this corpus.\n        '
        return [f.split('header.xml')[0] for f in self._fileids]

    def _view(self, filename, tags=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns a view specialised for use with particular corpus file.\n        '
        mode = kwargs.pop('mode', NKJPCorpusReader.WORDS_MODE)
        if mode is NKJPCorpusReader.WORDS_MODE:
            return NKJPCorpus_Morph_View(filename, tags=tags)
        elif mode is NKJPCorpusReader.SENTS_MODE:
            return NKJPCorpus_Segmentation_View(filename, tags=tags)
        elif mode is NKJPCorpusReader.HEADER_MODE:
            return NKJPCorpus_Header_View(filename, tags=tags)
        elif mode is NKJPCorpusReader.RAW_MODE:
            return NKJPCorpus_Text_View(filename, tags=tags, mode=NKJPCorpus_Text_View.RAW_MODE)
        else:
            raise NameError('No such mode!')

    def add_root(self, fileid):
        if False:
            return 10
        '\n        Add root if necessary to specified fileid.\n        '
        if self.root in fileid:
            return fileid
        return self.root + fileid

    @_parse_args
    def header(self, fileids=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns header(s) of specified fileids.\n        '
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.HEADER_MODE, **kwargs).handle_query() for fileid in fileids])

    @_parse_args
    def sents(self, fileids=None, **kwargs):
        if False:
            return 10
        '\n        Returns sentences in specified fileids.\n        '
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.SENTS_MODE, **kwargs).handle_query() for fileid in fileids])

    @_parse_args
    def words(self, fileids=None, **kwargs):
        if False:
            return 10
        '\n        Returns words in specified fileids.\n        '
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.WORDS_MODE, **kwargs).handle_query() for fileid in fileids])

    @_parse_args
    def tagged_words(self, fileids=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Call with specified tags as a list, e.g. tags=['subst', 'comp'].\n        Returns tagged words in specified fileids.\n        "
        tags = kwargs.pop('tags', [])
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.WORDS_MODE, tags=tags, **kwargs).handle_query() for fileid in fileids])

    @_parse_args
    def raw(self, fileids=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns words in specified fileids.\n        '
        return concat([self._view(self.add_root(fileid), mode=NKJPCorpusReader.RAW_MODE, **kwargs).handle_query() for fileid in fileids])

class NKJPCorpus_Header_View(XMLCorpusView):

    def __init__(self, filename, **kwargs):
        if False:
            while True:
                i = 10
        '\n        HEADER_MODE\n        A stream backed corpus view specialized for use with\n        header.xml files in NKJP corpus.\n        '
        self.tagspec = '.*/sourceDesc$'
        XMLCorpusView.__init__(self, filename + 'header.xml', self.tagspec)

    def handle_query(self):
        if False:
            print('Hello World!')
        self._open()
        header = []
        while True:
            segm = XMLCorpusView.read_block(self, self._stream)
            if len(segm) == 0:
                break
            header.extend(segm)
        self.close()
        return header

    def handle_elt(self, elt, context):
        if False:
            print('Hello World!')
        titles = elt.findall('bibl/title')
        title = []
        if titles:
            title = '\n'.join((title.text.strip() for title in titles))
        authors = elt.findall('bibl/author')
        author = []
        if authors:
            author = '\n'.join((author.text.strip() for author in authors))
        dates = elt.findall('bibl/date')
        date = []
        if dates:
            date = '\n'.join((date.text.strip() for date in dates))
        publishers = elt.findall('bibl/publisher')
        publisher = []
        if publishers:
            publisher = '\n'.join((publisher.text.strip() for publisher in publishers))
        idnos = elt.findall('bibl/idno')
        idno = []
        if idnos:
            idno = '\n'.join((idno.text.strip() for idno in idnos))
        notes = elt.findall('bibl/note')
        note = []
        if notes:
            note = '\n'.join((note.text.strip() for note in notes))
        return {'title': title, 'author': author, 'date': date, 'publisher': publisher, 'idno': idno, 'note': note}

class XML_Tool:
    """
    Helper class creating xml file to one without references to nkjp: namespace.
    That's needed because the XMLCorpusView assumes that one can find short substrings
    of XML that are valid XML, which is not true if a namespace is declared at top level
    """

    def __init__(self, root, filename):
        if False:
            return 10
        self.read_file = os.path.join(root, filename)
        self.write_file = tempfile.NamedTemporaryFile(delete=False)

    def build_preprocessed_file(self):
        if False:
            return 10
        try:
            fr = open(self.read_file)
            fw = self.write_file
            line = ' '
            while len(line):
                line = fr.readline()
                x = re.split('nkjp:[^ ]* ', line)
                ret = ' '.join(x)
                x = re.split('<nkjp:paren>', ret)
                ret = ' '.join(x)
                x = re.split('</nkjp:paren>', ret)
                ret = ' '.join(x)
                x = re.split('<choice>', ret)
                ret = ' '.join(x)
                x = re.split('</choice>', ret)
                ret = ' '.join(x)
                fw.write(ret)
            fr.close()
            fw.close()
            return self.write_file.name
        except Exception as e:
            self.remove_preprocessed_file()
            raise Exception from e

    def remove_preprocessed_file(self):
        if False:
            i = 10
            return i + 15
        os.remove(self.write_file.name)

class NKJPCorpus_Segmentation_View(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with
    ann_segmentation.xml files in NKJP corpus.
    """

    def __init__(self, filename, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.tagspec = '.*p/.*s'
        self.text_view = NKJPCorpus_Text_View(filename, mode=NKJPCorpus_Text_View.SENTS_MODE)
        self.text_view.handle_query()
        self.xml_tool = XML_Tool(filename, 'ann_segmentation.xml')
        XMLCorpusView.__init__(self, self.xml_tool.build_preprocessed_file(), self.tagspec)

    def get_segm_id(self, example_word):
        if False:
            print('Hello World!')
        return example_word.split('(')[1].split(',')[0]

    def get_sent_beg(self, beg_word):
        if False:
            print('Hello World!')
        return int(beg_word.split(',')[1])

    def get_sent_end(self, end_word):
        if False:
            for i in range(10):
                print('nop')
        splitted = end_word.split(')')[0].split(',')
        return int(splitted[1]) + int(splitted[2])

    def get_sentences(self, sent_segm):
        if False:
            return 10
        id = self.get_segm_id(sent_segm[0])
        segm = self.text_view.segm_dict[id]
        beg = self.get_sent_beg(sent_segm[0])
        end = self.get_sent_end(sent_segm[len(sent_segm) - 1])
        return segm[beg:end]

    def remove_choice(self, segm):
        if False:
            for i in range(10):
                print('nop')
        ret = []
        prev_txt_end = -1
        prev_txt_nr = -1
        for word in segm:
            txt_nr = self.get_segm_id(word)
            if self.get_sent_beg(word) > prev_txt_end - 1 or prev_txt_nr != txt_nr:
                ret.append(word)
                prev_txt_end = self.get_sent_end(word)
            prev_txt_nr = txt_nr
        return ret

    def handle_query(self):
        if False:
            while True:
                i = 10
        try:
            self._open()
            sentences = []
            while True:
                sent_segm = XMLCorpusView.read_block(self, self._stream)
                if len(sent_segm) == 0:
                    break
                for segm in sent_segm:
                    segm = self.remove_choice(segm)
                    sentences.append(self.get_sentences(segm))
            self.close()
            self.xml_tool.remove_preprocessed_file()
            return sentences
        except Exception as e:
            self.xml_tool.remove_preprocessed_file()
            raise Exception from e

    def handle_elt(self, elt, context):
        if False:
            i = 10
            return i + 15
        ret = []
        for seg in elt:
            ret.append(seg.get('corresp'))
        return ret

class NKJPCorpus_Text_View(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with
    text.xml files in NKJP corpus.
    """
    SENTS_MODE = 0
    RAW_MODE = 1

    def __init__(self, filename, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.mode = kwargs.pop('mode', 0)
        self.tagspec = '.*/div/ab'
        self.segm_dict = dict()
        self.xml_tool = XML_Tool(filename, 'text.xml')
        XMLCorpusView.__init__(self, self.xml_tool.build_preprocessed_file(), self.tagspec)

    def handle_query(self):
        if False:
            while True:
                i = 10
        try:
            self._open()
            x = self.read_block(self._stream)
            self.close()
            self.xml_tool.remove_preprocessed_file()
            return x
        except Exception as e:
            self.xml_tool.remove_preprocessed_file()
            raise Exception from e

    def read_block(self, stream, tagspec=None, elt_handler=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns text as a list of sentences.\n        '
        txt = []
        while True:
            segm = XMLCorpusView.read_block(self, stream)
            if len(segm) == 0:
                break
            for part in segm:
                txt.append(part)
        return [' '.join([segm for segm in txt])]

    def get_segm_id(self, elt):
        if False:
            for i in range(10):
                print('nop')
        for attr in elt.attrib:
            if attr.endswith('id'):
                return elt.get(attr)

    def handle_elt(self, elt, context):
        if False:
            while True:
                i = 10
        if self.mode is NKJPCorpus_Text_View.SENTS_MODE:
            self.segm_dict[self.get_segm_id(elt)] = elt.text
        return elt.text

class NKJPCorpus_Morph_View(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with
    ann_morphosyntax.xml files in NKJP corpus.
    """

    def __init__(self, filename, **kwargs):
        if False:
            i = 10
            return i + 15
        self.tags = kwargs.pop('tags', None)
        self.tagspec = '.*/seg/fs'
        self.xml_tool = XML_Tool(filename, 'ann_morphosyntax.xml')
        XMLCorpusView.__init__(self, self.xml_tool.build_preprocessed_file(), self.tagspec)

    def handle_query(self):
        if False:
            return 10
        try:
            self._open()
            words = []
            while True:
                segm = XMLCorpusView.read_block(self, self._stream)
                if len(segm) == 0:
                    break
                for part in segm:
                    if part is not None:
                        words.append(part)
            self.close()
            self.xml_tool.remove_preprocessed_file()
            return words
        except Exception as e:
            self.xml_tool.remove_preprocessed_file()
            raise Exception from e

    def handle_elt(self, elt, context):
        if False:
            while True:
                i = 10
        word = ''
        flag = False
        is_not_interp = True
        if self.tags is None:
            flag = True
        for child in elt:
            if 'name' in child.keys() and child.attrib['name'] == 'orth':
                for symbol in child:
                    if symbol.tag == 'string':
                        word = symbol.text
            elif 'name' in child.keys() and child.attrib['name'] == 'interps':
                for symbol in child:
                    if 'type' in symbol.keys() and symbol.attrib['type'] == 'lex':
                        for symbol2 in symbol:
                            if 'name' in symbol2.keys() and symbol2.attrib['name'] == 'ctag':
                                for symbol3 in symbol2:
                                    if 'value' in symbol3.keys() and self.tags is not None and (symbol3.attrib['value'] in self.tags):
                                        flag = True
                                    elif 'value' in symbol3.keys() and symbol3.attrib['value'] == 'interp':
                                        is_not_interp = False
        if flag and is_not_interp:
            return word