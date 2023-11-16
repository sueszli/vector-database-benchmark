"""
Corpus reader for the XML version of the CHILDES corpus.
"""
__docformat__ = 'epytext en'
import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
NS = 'http://www.talkbank.org/ns/talkbank'

class CHILDESCorpusReader(XMLCorpusReader):
    """
    Corpus reader for the XML version of the CHILDES corpus.
    The CHILDES corpus is available at ``https://childes.talkbank.org/``. The XML
    version of CHILDES is located at ``https://childes.talkbank.org/data-xml/``.
    Copy the needed parts of the CHILDES XML corpus into the NLTK data directory
    (``nltk_data/corpora/CHILDES/``).

    For access to the file text use the usual nltk functions,
    ``words()``, ``sents()``, ``tagged_words()`` and ``tagged_sents()``.
    """

    def __init__(self, root, fileids, lazy=True):
        if False:
            return 10
        XMLCorpusReader.__init__(self, root, fileids)
        self._lazy = lazy

    def words(self, fileids=None, speaker='ALL', stem=False, relation=False, strip_space=True, replace=False):
        if False:
            print('Hello World!')
        "\n        :return: the given file(s) as a list of words\n        :rtype: list(str)\n\n        :param speaker: If specified, select specific speaker(s) defined\n            in the corpus. Default is 'ALL' (all participants). Common choices\n            are 'CHI' (the child), 'MOT' (mother), ['CHI','MOT'] (exclude\n            researchers)\n        :param stem: If true, then use word stems instead of word strings.\n        :param relation: If true, then return tuples of (stem, index,\n            dependent_index)\n        :param strip_space: If true, then strip trailing spaces from word\n            tokens. Otherwise, leave the spaces on the tokens.\n        :param replace: If true, then use the replaced (intended) word instead\n            of the original word (e.g., 'wat' will be replaced with 'watch')\n        "
        sent = None
        pos = False
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace) for fileid in self.abspaths(fileids)]
        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))

    def tagged_words(self, fileids=None, speaker='ALL', stem=False, relation=False, strip_space=True, replace=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        :return: the given file(s) as a list of tagged\n            words and punctuation symbols, encoded as tuples\n            ``(word,tag)``.\n        :rtype: list(tuple(str,str))\n\n        :param speaker: If specified, select specific speaker(s) defined\n            in the corpus. Default is 'ALL' (all participants). Common choices\n            are 'CHI' (the child), 'MOT' (mother), ['CHI','MOT'] (exclude\n            researchers)\n        :param stem: If true, then use word stems instead of word strings.\n        :param relation: If true, then return tuples of (stem, index,\n            dependent_index)\n        :param strip_space: If true, then strip trailing spaces from word\n            tokens. Otherwise, leave the spaces on the tokens.\n        :param replace: If true, then use the replaced (intended) word instead\n            of the original word (e.g., 'wat' will be replaced with 'watch')\n        "
        sent = None
        pos = True
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace) for fileid in self.abspaths(fileids)]
        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))

    def sents(self, fileids=None, speaker='ALL', stem=False, relation=None, strip_space=True, replace=False):
        if False:
            while True:
                i = 10
        "\n        :return: the given file(s) as a list of sentences or utterances, each\n            encoded as a list of word strings.\n        :rtype: list(list(str))\n\n        :param speaker: If specified, select specific speaker(s) defined\n            in the corpus. Default is 'ALL' (all participants). Common choices\n            are 'CHI' (the child), 'MOT' (mother), ['CHI','MOT'] (exclude\n            researchers)\n        :param stem: If true, then use word stems instead of word strings.\n        :param relation: If true, then return tuples of ``(str,pos,relation_list)``.\n            If there is manually-annotated relation info, it will return\n            tuples of ``(str,pos,test_relation_list,str,pos,gold_relation_list)``\n        :param strip_space: If true, then strip trailing spaces from word\n            tokens. Otherwise, leave the spaces on the tokens.\n        :param replace: If true, then use the replaced (intended) word instead\n            of the original word (e.g., 'wat' will be replaced with 'watch')\n        "
        sent = True
        pos = False
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace) for fileid in self.abspaths(fileids)]
        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))

    def tagged_sents(self, fileids=None, speaker='ALL', stem=False, relation=None, strip_space=True, replace=False):
        if False:
            while True:
                i = 10
        "\n        :return: the given file(s) as a list of\n            sentences, each encoded as a list of ``(word,tag)`` tuples.\n        :rtype: list(list(tuple(str,str)))\n\n        :param speaker: If specified, select specific speaker(s) defined\n            in the corpus. Default is 'ALL' (all participants). Common choices\n            are 'CHI' (the child), 'MOT' (mother), ['CHI','MOT'] (exclude\n            researchers)\n        :param stem: If true, then use word stems instead of word strings.\n        :param relation: If true, then return tuples of ``(str,pos,relation_list)``.\n            If there is manually-annotated relation info, it will return\n            tuples of ``(str,pos,test_relation_list,str,pos,gold_relation_list)``\n        :param strip_space: If true, then strip trailing spaces from word\n            tokens. Otherwise, leave the spaces on the tokens.\n        :param replace: If true, then use the replaced (intended) word instead\n            of the original word (e.g., 'wat' will be replaced with 'watch')\n        "
        sent = True
        pos = True
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace) for fileid in self.abspaths(fileids)]
        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))

    def corpus(self, fileids=None):
        if False:
            print('Hello World!')
        '\n        :return: the given file(s) as a dict of ``(corpus_property_key, value)``\n        :rtype: list(dict)\n        '
        if not self._lazy:
            return [self._get_corpus(fileid) for fileid in self.abspaths(fileids)]
        return LazyMap(self._get_corpus, self.abspaths(fileids))

    def _get_corpus(self, fileid):
        if False:
            print('Hello World!')
        results = dict()
        xmldoc = ElementTree.parse(fileid).getroot()
        for (key, value) in xmldoc.items():
            results[key] = value
        return results

    def participants(self, fileids=None):
        if False:
            i = 10
            return i + 15
        '\n        :return: the given file(s) as a dict of\n            ``(participant_property_key, value)``\n        :rtype: list(dict)\n        '
        if not self._lazy:
            return [self._get_participants(fileid) for fileid in self.abspaths(fileids)]
        return LazyMap(self._get_participants, self.abspaths(fileids))

    def _get_participants(self, fileid):
        if False:
            i = 10
            return i + 15

        def dictOfDicts():
            if False:
                return 10
            return defaultdict(dictOfDicts)
        xmldoc = ElementTree.parse(fileid).getroot()
        pat = dictOfDicts()
        for participant in xmldoc.findall(f'.//{{{NS}}}Participants/{{{NS}}}participant'):
            for (key, value) in participant.items():
                pat[participant.get('id')][key] = value
        return pat

    def age(self, fileids=None, speaker='CHI', month=False):
        if False:
            i = 10
            return i + 15
        '\n        :return: the given file(s) as string or int\n        :rtype: list or int\n\n        :param month: If true, return months instead of year-month-date\n        '
        if not self._lazy:
            return [self._get_age(fileid, speaker, month) for fileid in self.abspaths(fileids)]
        get_age = lambda fileid: self._get_age(fileid, speaker, month)
        return LazyMap(get_age, self.abspaths(fileids))

    def _get_age(self, fileid, speaker, month):
        if False:
            for i in range(10):
                print('nop')
        xmldoc = ElementTree.parse(fileid).getroot()
        for pat in xmldoc.findall(f'.//{{{NS}}}Participants/{{{NS}}}participant'):
            try:
                if pat.get('id') == speaker:
                    age = pat.get('age')
                    if month:
                        age = self.convert_age(age)
                    return age
            except (TypeError, AttributeError) as e:
                return None

    def convert_age(self, age_year):
        if False:
            for i in range(10):
                print('nop')
        'Caclculate age in months from a string in CHILDES format'
        m = re.match('P(\\d+)Y(\\d+)M?(\\d?\\d?)D?', age_year)
        age_month = int(m.group(1)) * 12 + int(m.group(2))
        try:
            if int(m.group(3)) > 15:
                age_month += 1
        except ValueError as e:
            pass
        return age_month

    def MLU(self, fileids=None, speaker='CHI'):
        if False:
            return 10
        '\n        :return: the given file(s) as a floating number\n        :rtype: list(float)\n        '
        if not self._lazy:
            return [self._getMLU(fileid, speaker=speaker) for fileid in self.abspaths(fileids)]
        get_MLU = lambda fileid: self._getMLU(fileid, speaker=speaker)
        return LazyMap(get_MLU, self.abspaths(fileids))

    def _getMLU(self, fileid, speaker):
        if False:
            i = 10
            return i + 15
        sents = self._get_words(fileid, speaker=speaker, sent=True, stem=True, relation=False, pos=True, strip_space=True, replace=True)
        results = []
        lastSent = []
        numFillers = 0
        sentDiscount = 0
        for sent in sents:
            posList = [pos for (word, pos) in sent]
            if any((pos == 'unk' for pos in posList)):
                continue
            elif sent == []:
                continue
            elif sent == lastSent:
                continue
            else:
                results.append([word for (word, pos) in sent])
                if len({'co', None}.intersection(posList)) > 0:
                    numFillers += posList.count('co')
                    numFillers += posList.count(None)
                    sentDiscount += 1
            lastSent = sent
        try:
            thisWordList = flatten(results)
            numWords = len(flatten([word.split('-') for word in thisWordList])) - numFillers
            numSents = len(results) - sentDiscount
            mlu = numWords / numSents
        except ZeroDivisionError:
            mlu = 0
        return mlu

    def _get_words(self, fileid, speaker, sent, stem, relation, pos, strip_space, replace):
        if False:
            return 10
        if isinstance(speaker, str) and speaker != 'ALL':
            speaker = [speaker]
        xmldoc = ElementTree.parse(fileid).getroot()
        results = []
        for xmlsent in xmldoc.findall('.//{%s}u' % NS):
            sents = []
            if speaker == 'ALL' or xmlsent.get('who') in speaker:
                for xmlword in xmlsent.findall('.//{%s}w' % NS):
                    infl = None
                    suffixStem = None
                    suffixTag = None
                    if replace and xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}replacement'):
                        xmlword = xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}replacement/{{{NS}}}w')
                    elif replace and xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}wk'):
                        xmlword = xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}wk')
                    if xmlword.text:
                        word = xmlword.text
                    else:
                        word = ''
                    if strip_space:
                        word = word.strip()
                    if relation or stem:
                        try:
                            xmlstem = xmlword.find('.//{%s}stem' % NS)
                            word = xmlstem.text
                        except AttributeError as e:
                            pass
                        try:
                            xmlinfl = xmlword.find(f'.//{{{NS}}}mor/{{{NS}}}mw/{{{NS}}}mk')
                            word += '-' + xmlinfl.text
                        except:
                            pass
                        try:
                            xmlsuffix = xmlword.find('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}stem' % (NS, NS, NS, NS))
                            suffixStem = xmlsuffix.text
                        except AttributeError:
                            suffixStem = ''
                        if suffixStem:
                            word += '~' + suffixStem
                    if relation or pos:
                        try:
                            xmlpos = xmlword.findall('.//{%s}c' % NS)
                            xmlpos2 = xmlword.findall('.//{%s}s' % NS)
                            if xmlpos2 != []:
                                tag = xmlpos[0].text + ':' + xmlpos2[0].text
                            else:
                                tag = xmlpos[0].text
                        except (AttributeError, IndexError) as e:
                            tag = ''
                        try:
                            xmlsuffixpos = xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}pos/{%s}c' % (NS, NS, NS, NS, NS))
                            xmlsuffixpos2 = xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}pos/{%s}s' % (NS, NS, NS, NS, NS))
                            if xmlsuffixpos2:
                                suffixTag = xmlsuffixpos[0].text + ':' + xmlsuffixpos2[0].text
                            else:
                                suffixTag = xmlsuffixpos[0].text
                        except:
                            pass
                        if suffixTag:
                            tag += '~' + suffixTag
                        word = (word, tag)
                    if relation == True:
                        for xmlstem_rel in xmlword.findall(f'.//{{{NS}}}mor/{{{NS}}}gra'):
                            if not xmlstem_rel.get('type') == 'grt':
                                word = (word[0], word[1], xmlstem_rel.get('index') + '|' + xmlstem_rel.get('head') + '|' + xmlstem_rel.get('relation'))
                            else:
                                word = (word[0], word[1], word[2], word[0], word[1], xmlstem_rel.get('index') + '|' + xmlstem_rel.get('head') + '|' + xmlstem_rel.get('relation'))
                        try:
                            for xmlpost_rel in xmlword.findall(f'.//{{{NS}}}mor/{{{NS}}}mor-post/{{{NS}}}gra'):
                                if not xmlpost_rel.get('type') == 'grt':
                                    suffixStem = (suffixStem[0], suffixStem[1], xmlpost_rel.get('index') + '|' + xmlpost_rel.get('head') + '|' + xmlpost_rel.get('relation'))
                                else:
                                    suffixStem = (suffixStem[0], suffixStem[1], suffixStem[2], suffixStem[0], suffixStem[1], xmlpost_rel.get('index') + '|' + xmlpost_rel.get('head') + '|' + xmlpost_rel.get('relation'))
                        except:
                            pass
                    sents.append(word)
                if sent or relation:
                    results.append(sents)
                else:
                    results.extend(sents)
        return LazyMap(lambda x: x, results)
    "\n    The base URL for viewing files on the childes website. This\n    shouldn't need to be changed, unless CHILDES changes the configuration\n    of their server or unless the user sets up their own corpus webserver.\n    "
    childes_url_base = 'https://childes.talkbank.org/browser/index.php?url='

    def webview_file(self, fileid, urlbase=None):
        if False:
            return 10
        'Map a corpus file to its web version on the CHILDES website,\n        and open it in a web browser.\n\n        The complete URL to be used is:\n            childes.childes_url_base + urlbase + fileid.replace(\'.xml\', \'.cha\')\n\n        If no urlbase is passed, we try to calculate it.  This\n        requires that the childes corpus was set up to mirror the\n        folder hierarchy under childes.psy.cmu.edu/data-xml/, e.g.:\n        nltk_data/corpora/childes/Eng-USA/Cornell/??? or\n        nltk_data/corpora/childes/Romance/Spanish/Aguirre/???\n\n        The function first looks (as a special case) if "Eng-USA" is\n        on the path consisting of <corpus root>+fileid; then if\n        "childes", possibly followed by "data-xml", appears. If neither\n        one is found, we use the unmodified fileid and hope for the best.\n        If this is not right, specify urlbase explicitly, e.g., if the\n        corpus root points to the Cornell folder, urlbase=\'Eng-USA/Cornell\'.\n        '
        import webbrowser
        if urlbase:
            path = urlbase + '/' + fileid
        else:
            full = self.root + '/' + fileid
            full = re.sub('\\\\', '/', full)
            if '/childes/' in full.lower():
                path = re.findall('(?i)/childes(?:/data-xml)?/(.*)\\.xml', full)[0]
            elif 'eng-usa' in full.lower():
                path = 'Eng-USA/' + re.findall('/(?i)Eng-USA/(.*)\\.xml', full)[0]
            else:
                path = fileid
        if path.endswith('.xml'):
            path = path[:-4]
        if not path.endswith('.cha'):
            path = path + '.cha'
        url = self.childes_url_base + path
        webbrowser.open_new_tab(url)
        print('Opening in browser:', url)

def demo(corpus_root=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    The CHILDES corpus should be manually downloaded and saved\n    to ``[NLTK_Data_Dir]/corpora/childes/``\n    '
    if not corpus_root:
        from nltk.data import find
        corpus_root = find('corpora/childes/data-xml/Eng-USA/')
    try:
        childes = CHILDESCorpusReader(corpus_root, '.*.xml')
        for file in childes.fileids()[:5]:
            corpus = ''
            corpus_id = ''
            for (key, value) in childes.corpus(file)[0].items():
                if key == 'Corpus':
                    corpus = value
                if key == 'Id':
                    corpus_id = value
            print('Reading', corpus, corpus_id, ' .....')
            print('words:', childes.words(file)[:7], '...')
            print('words with replaced words:', childes.words(file, replace=True)[:7], ' ...')
            print('words with pos tags:', childes.tagged_words(file)[:7], ' ...')
            print('words (only MOT):', childes.words(file, speaker='MOT')[:7], '...')
            print('words (only CHI):', childes.words(file, speaker='CHI')[:7], '...')
            print('stemmed words:', childes.words(file, stem=True)[:7], ' ...')
            print('words with relations and pos-tag:', childes.words(file, relation=True)[:5], ' ...')
            print('sentence:', childes.sents(file)[:2], ' ...')
            for (participant, values) in childes.participants(file)[0].items():
                for (key, value) in values.items():
                    print('\tparticipant', participant, key, ':', value)
            print('num of sent:', len(childes.sents(file)))
            print('num of morphemes:', len(childes.words(file, stem=True)))
            print('age:', childes.age(file))
            print('age in month:', childes.age(file, month=True))
            print('MLU:', childes.MLU(file))
            print()
    except LookupError as e:
        print('The CHILDES corpus, or the parts you need, should be manually\n        downloaded from https://childes.talkbank.org/data-xml/ and saved at\n        [NLTK_Data_Dir]/corpora/childes/\n            Alternately, you can call the demo with the path to a portion of the CHILDES corpus, e.g.:\n        demo(\'/path/to/childes/data-xml/Eng-USA/")\n        ')
if __name__ == '__main__':
    demo()