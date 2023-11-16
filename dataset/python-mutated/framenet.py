"""
Corpus reader for the FrameNet 1.7 lexicon and corpus.
"""
import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
__docformat__ = 'epytext en'

def mimic_wrap(lines, wrap_at=65, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Wrap the first of 'lines' with textwrap and the remaining lines at exactly the same\n    positions as the first.\n    "
    l0 = textwrap.fill(lines[0], wrap_at, drop_whitespace=False).split('\n')
    yield l0

    def _(line):
        if False:
            i = 10
            return i + 15
        il0 = 0
        while line and il0 < len(l0) - 1:
            yield line[:len(l0[il0])]
            line = line[len(l0[il0]):]
            il0 += 1
        if line:
            yield from textwrap.fill(line, wrap_at, drop_whitespace=False).split('\n')
    for l in lines[1:]:
        yield list(_(l))

def _pretty_longstring(defstr, prefix='', wrap_at=65):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function for pretty-printing a long string.\n\n    :param defstr: The string to be printed.\n    :type defstr: str\n    :return: A nicely formatted string representation of the long string.\n    :rtype: str\n    '
    outstr = ''
    for line in textwrap.fill(defstr, wrap_at).split('\n'):
        outstr += prefix + line + '\n'
    return outstr

def _pretty_any(obj):
    if False:
        print('Hello World!')
    '\n    Helper function for pretty-printing any AttrDict object.\n\n    :param obj: The obj to be printed.\n    :type obj: AttrDict\n    :return: A nicely formatted string representation of the AttrDict object.\n    :rtype: str\n    '
    outstr = ''
    for k in obj:
        if isinstance(obj[k], str) and len(obj[k]) > 65:
            outstr += f'[{k}]\n'
            outstr += '{}'.format(_pretty_longstring(obj[k], prefix='  '))
            outstr += '\n'
        else:
            outstr += f'[{k}] {obj[k]}\n'
    return outstr

def _pretty_semtype(st):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function for pretty-printing a semantic type.\n\n    :param st: The semantic type to be printed.\n    :type st: AttrDict\n    :return: A nicely formatted string representation of the semantic type.\n    :rtype: str\n    '
    semkeys = st.keys()
    if len(semkeys) == 1:
        return '<None>'
    outstr = ''
    outstr += 'semantic type ({0.ID}): {0.name}\n'.format(st)
    if 'abbrev' in semkeys:
        outstr += f'[abbrev] {st.abbrev}\n'
    if 'definition' in semkeys:
        outstr += '[definition]\n'
        outstr += _pretty_longstring(st.definition, '  ')
    outstr += f'[rootType] {st.rootType.name}({st.rootType.ID})\n'
    if st.superType is None:
        outstr += '[superType] <None>\n'
    else:
        outstr += f'[superType] {st.superType.name}({st.superType.ID})\n'
    outstr += f'[subTypes] {len(st.subTypes)} subtypes\n'
    outstr += '  ' + ', '.join((f'{x.name}({x.ID})' for x in st.subTypes)) + '\n' * (len(st.subTypes) > 0)
    return outstr

def _pretty_frame_relation_type(freltyp):
    if False:
        return 10
    '\n    Helper function for pretty-printing a frame relation type.\n\n    :param freltyp: The frame relation type to be printed.\n    :type freltyp: AttrDict\n    :return: A nicely formatted string representation of the frame relation type.\n    :rtype: str\n    '
    outstr = '<frame relation type ({0.ID}): {0.superFrameName} -- {0.name} -> {0.subFrameName}>'.format(freltyp)
    return outstr

def _pretty_frame_relation(frel):
    if False:
        print('Hello World!')
    '\n    Helper function for pretty-printing a frame relation.\n\n    :param frel: The frame relation to be printed.\n    :type frel: AttrDict\n    :return: A nicely formatted string representation of the frame relation.\n    :rtype: str\n    '
    outstr = '<{0.type.superFrameName}={0.superFrameName} -- {0.type.name} -> {0.type.subFrameName}={0.subFrameName}>'.format(frel)
    return outstr

def _pretty_fe_relation(ferel):
    if False:
        while True:
            i = 10
    '\n    Helper function for pretty-printing an FE relation.\n\n    :param ferel: The FE relation to be printed.\n    :type ferel: AttrDict\n    :return: A nicely formatted string representation of the FE relation.\n    :rtype: str\n    '
    outstr = '<{0.type.superFrameName}={0.frameRelation.superFrameName}.{0.superFEName} -- {0.type.name} -> {0.type.subFrameName}={0.frameRelation.subFrameName}.{0.subFEName}>'.format(ferel)
    return outstr

def _pretty_lu(lu):
    if False:
        print('Hello World!')
    '\n    Helper function for pretty-printing a lexical unit.\n\n    :param lu: The lu to be printed.\n    :type lu: AttrDict\n    :return: A nicely formatted string representation of the lexical unit.\n    :rtype: str\n    '
    lukeys = lu.keys()
    outstr = ''
    outstr += 'lexical unit ({0.ID}): {0.name}\n\n'.format(lu)
    if 'definition' in lukeys:
        outstr += '[definition]\n'
        outstr += _pretty_longstring(lu.definition, '  ')
    if 'frame' in lukeys:
        outstr += f'\n[frame] {lu.frame.name}({lu.frame.ID})\n'
    if 'incorporatedFE' in lukeys:
        outstr += f'\n[incorporatedFE] {lu.incorporatedFE}\n'
    if 'POS' in lukeys:
        outstr += f'\n[POS] {lu.POS}\n'
    if 'status' in lukeys:
        outstr += f'\n[status] {lu.status}\n'
    if 'totalAnnotated' in lukeys:
        outstr += f'\n[totalAnnotated] {lu.totalAnnotated} annotated examples\n'
    if 'lexemes' in lukeys:
        outstr += '\n[lexemes] {}\n'.format(' '.join((f'{lex.name}/{lex.POS}' for lex in lu.lexemes)))
    if 'semTypes' in lukeys:
        outstr += f'\n[semTypes] {len(lu.semTypes)} semantic types\n'
        outstr += '  ' * (len(lu.semTypes) > 0) + ', '.join((f'{x.name}({x.ID})' for x in lu.semTypes)) + '\n' * (len(lu.semTypes) > 0)
    if 'URL' in lukeys:
        outstr += f'\n[URL] {lu.URL}\n'
    if 'subCorpus' in lukeys:
        subc = [x.name for x in lu.subCorpus]
        outstr += f'\n[subCorpus] {len(lu.subCorpus)} subcorpora\n'
        for line in textwrap.fill(', '.join(sorted(subc)), 60).split('\n'):
            outstr += f'  {line}\n'
    if 'exemplars' in lukeys:
        outstr += '\n[exemplars] {} sentences across all subcorpora\n'.format(len(lu.exemplars))
    return outstr

def _pretty_exemplars(exemplars, lu):
    if False:
        print('Hello World!')
    '\n    Helper function for pretty-printing a list of exemplar sentences for a lexical unit.\n\n    :param sent: The list of exemplar sentences to be printed.\n    :type sent: list(AttrDict)\n    :return: An index of the text of the exemplar sentences.\n    :rtype: str\n    '
    outstr = ''
    outstr += 'exemplar sentences for {0.name} in {0.frame.name}:\n\n'.format(lu)
    for (i, sent) in enumerate(exemplars):
        outstr += f'[{i}] {sent.text}\n'
    outstr += '\n'
    return outstr

def _pretty_fulltext_sentences(sents):
    if False:
        i = 10
        return i + 15
    '\n    Helper function for pretty-printing a list of annotated sentences for a full-text document.\n\n    :param sent: The list of sentences to be printed.\n    :type sent: list(AttrDict)\n    :return: An index of the text of the sentences.\n    :rtype: str\n    '
    outstr = ''
    outstr += 'full-text document ({0.ID}) {0.name}:\n\n'.format(sents)
    outstr += '[corpid] {0.corpid}\n[corpname] {0.corpname}\n[description] {0.description}\n[URL] {0.URL}\n\n'.format(sents)
    outstr += f'[sentence]\n'
    for (i, sent) in enumerate(sents.sentence):
        outstr += f'[{i}] {sent.text}\n'
    outstr += '\n'
    return outstr

def _pretty_fulltext_sentence(sent):
    if False:
        return 10
    '\n    Helper function for pretty-printing an annotated sentence from a full-text document.\n\n    :param sent: The sentence to be printed.\n    :type sent: list(AttrDict)\n    :return: The text of the sentence with annotation set indices on frame targets.\n    :rtype: str\n    '
    outstr = ''
    outstr += 'full-text sentence ({0.ID}) in {1}:\n\n'.format(sent, sent.doc.get('name', sent.doc.description))
    outstr += f'\n[POS] {len(sent.POS)} tags\n'
    outstr += f'\n[POS_tagset] {sent.POS_tagset}\n\n'
    outstr += '[text] + [annotationSet]\n\n'
    outstr += sent._ascii()
    outstr += '\n'
    return outstr

def _pretty_pos(aset):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function for pretty-printing a sentence with its POS tags.\n\n    :param aset: The POS annotation set of the sentence to be printed.\n    :type sent: list(AttrDict)\n    :return: The text of the sentence and its POS tags.\n    :rtype: str\n    '
    outstr = ''
    outstr += 'POS annotation set ({0.ID}) {0.POS_tagset} in sentence {0.sent.ID}:\n\n'.format(aset)
    overt = sorted(aset.POS)
    sent = aset.sent
    s0 = sent.text
    s1 = ''
    s2 = ''
    i = 0
    adjust = 0
    for (j, k, lbl) in overt:
        assert j >= i, ('Overlapping targets?', (j, k, lbl))
        s1 += ' ' * (j - i) + '-' * (k - j)
        if len(lbl) > k - j:
            amt = len(lbl) - (k - j)
            s0 = s0[:k + adjust] + '~' * amt + s0[k + adjust:]
            s1 = s1[:k + adjust] + ' ' * amt + s1[k + adjust:]
            adjust += amt
        s2 += ' ' * (j - i) + lbl.ljust(k - j)
        i = k
    long_lines = [s0, s1, s2]
    outstr += '\n\n'.join(map('\n'.join, zip_longest(*mimic_wrap(long_lines), fillvalue=' '))).replace('~', ' ')
    outstr += '\n'
    return outstr

def _pretty_annotation(sent, aset_level=False):
    if False:
        return 10
    "\n    Helper function for pretty-printing an exemplar sentence for a lexical unit.\n\n    :param sent: An annotation set or exemplar sentence to be printed.\n    :param aset_level: If True, 'sent' is actually an annotation set within a sentence.\n    :type sent: AttrDict\n    :return: A nicely formatted string representation of the exemplar sentence\n    with its target, frame, and FE annotations.\n    :rtype: str\n    "
    sentkeys = sent.keys()
    outstr = 'annotation set' if aset_level else 'exemplar sentence'
    outstr += f' ({sent.ID}):\n'
    if aset_level:
        outstr += f'\n[status] {sent.status}\n'
    for k in ('corpID', 'docID', 'paragNo', 'sentNo', 'aPos'):
        if k in sentkeys:
            outstr += f'[{k}] {sent[k]}\n'
    outstr += '\n[LU] ({0.ID}) {0.name} in {0.frame.name}\n'.format(sent.LU) if sent.LU else '\n[LU] Not found!'
    outstr += '\n[frame] ({0.ID}) {0.name}\n'.format(sent.frame)
    if not aset_level:
        outstr += '\n[annotationSet] {} annotation sets\n'.format(len(sent.annotationSet))
        outstr += f'\n[POS] {len(sent.POS)} tags\n'
        outstr += f'\n[POS_tagset] {sent.POS_tagset}\n'
    outstr += '\n[GF] {} relation{}\n'.format(len(sent.GF), 's' if len(sent.GF) != 1 else '')
    outstr += '\n[PT] {} phrase{}\n'.format(len(sent.PT), 's' if len(sent.PT) != 1 else '')
    "\n    Special Layers\n    --------------\n\n    The 'NER' layer contains, for some of the data, named entity labels.\n\n    The 'WSL' (word status layer) contains, for some of the data,\n    spans which should not in principle be considered targets (NT).\n\n    The 'Other' layer records relative clause constructions (Rel=relativizer, Ant=antecedent),\n    pleonastic 'it' (Null), and existential 'there' (Exist).\n    On occasion they are duplicated by accident (e.g., annotationSet 1467275 in lu6700.xml).\n\n    The 'Sent' layer appears to contain labels that the annotator has flagged the\n    sentence with for their convenience: values include\n    'sense1', 'sense2', 'sense3', etc.;\n    'Blend', 'Canonical', 'Idiom', 'Metaphor', 'Special-Sent',\n    'keepS', 'deleteS', 'reexamine'\n    (sometimes they are duplicated for no apparent reason).\n\n    The POS-specific layers may contain the following kinds of spans:\n    Asp (aspectual particle), Non-Asp (non-aspectual particle),\n    Cop (copula), Supp (support), Ctrlr (controller),\n    Gov (governor), X. Gov and X always cooccur.\n\n    >>> from nltk.corpus import framenet as fn\n    >>> def f(luRE, lyr, ignore=set()):\n    ...   for i,ex in enumerate(fn.exemplars(luRE)):\n    ...     if lyr in ex and ex[lyr] and set(zip(*ex[lyr])[2]) - ignore:\n    ...       print(i,ex[lyr])\n\n    - Verb: Asp, Non-Asp\n    - Noun: Cop, Supp, Ctrlr, Gov, X\n    - Adj: Cop, Supp, Ctrlr, Gov, X\n    - Prep: Cop, Supp, Ctrlr\n    - Adv: Ctrlr\n    - Scon: (none)\n    - Art: (none)\n    "
    for lyr in ('NER', 'WSL', 'Other', 'Sent'):
        if lyr in sent and sent[lyr]:
            outstr += '\n[{}] {} entr{}\n'.format(lyr, len(sent[lyr]), 'ies' if len(sent[lyr]) != 1 else 'y')
    outstr += '\n[text] + [Target] + [FE]'
    for lyr in ('Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art'):
        if lyr in sent and sent[lyr]:
            outstr += f' + [{lyr}]'
    if 'FE2' in sentkeys:
        outstr += ' + [FE2]'
        if 'FE3' in sentkeys:
            outstr += ' + [FE3]'
    outstr += '\n\n'
    outstr += sent._ascii()
    outstr += '\n'
    return outstr

def _annotation_ascii(sent):
    if False:
        return 10
    "\n    Given a sentence or FE annotation set, construct the width-limited string showing\n    an ASCII visualization of the sentence's annotations, calling either\n    _annotation_ascii_frames() or _annotation_ascii_FEs() as appropriate.\n    This will be attached as a method to appropriate AttrDict instances\n    and called in the full pretty-printing of the instance.\n    "
    if sent._type == 'fulltext_sentence' or ('annotationSet' in sent and len(sent.annotationSet) > 2):
        return _annotation_ascii_frames(sent)
    else:
        return _annotation_ascii_FEs(sent)

def _annotation_ascii_frames(sent):
    if False:
        for i in range(10):
            print('nop')
    '\n    ASCII string rendering of the sentence along with its targets and frame names.\n    Called for all full-text sentences, as well as the few LU sentences with multiple\n    targets (e.g., fn.lu(6412).exemplars[82] has two want.v targets).\n    Line-wrapped to limit the display width.\n    '
    overt = []
    for (a, aset) in enumerate(sent.annotationSet[1:]):
        for (j, k) in aset.Target:
            indexS = f'[{a + 1}]'
            if aset.status == 'UNANN' or aset.LU.status == 'Problem':
                indexS += ' '
                if aset.status == 'UNANN':
                    indexS += '!'
                if aset.LU.status == 'Problem':
                    indexS += '?'
            overt.append((j, k, aset.LU.frame.name, indexS))
    overt = sorted(overt)
    duplicates = set()
    for (o, (j, k, fname, asetIndex)) in enumerate(overt):
        if o > 0 and j <= overt[o - 1][1]:
            if overt[o - 1][:2] == (j, k) and overt[o - 1][2] == fname:
                combinedIndex = overt[o - 1][3] + asetIndex
                combinedIndex = combinedIndex.replace(' !', '! ').replace(' ?', '? ')
                overt[o - 1] = overt[o - 1][:3] + (combinedIndex,)
                duplicates.add(o)
            else:
                s = sent.text
                for (j, k, fname, asetIndex) in overt:
                    s += '\n' + asetIndex + ' ' + sent.text[j:k] + ' :: ' + fname
                s += '\n(Unable to display sentence with targets marked inline due to overlap)'
                return s
    for o in reversed(sorted(duplicates)):
        del overt[o]
    s0 = sent.text
    s1 = ''
    s11 = ''
    s2 = ''
    i = 0
    adjust = 0
    fAbbrevs = OrderedDict()
    for (j, k, fname, asetIndex) in overt:
        if not j >= i:
            assert j >= i, ('Overlapping targets?' + (' UNANN' if any((aset.status == 'UNANN' for aset in sent.annotationSet[1:])) else ''), (j, k, asetIndex))
        s1 += ' ' * (j - i) + '*' * (k - j)
        short = fname[:k - j]
        if k - j < len(fname):
            r = 0
            while short in fAbbrevs:
                if fAbbrevs[short] == fname:
                    break
                r += 1
                short = fname[:k - j - 1] + str(r)
            else:
                fAbbrevs[short] = fname
        s11 += ' ' * (j - i) + short.ljust(k - j)
        if len(asetIndex) > k - j:
            amt = len(asetIndex) - (k - j)
            s0 = s0[:k + adjust] + '~' * amt + s0[k + adjust:]
            s1 = s1[:k + adjust] + ' ' * amt + s1[k + adjust:]
            s11 = s11[:k + adjust] + ' ' * amt + s11[k + adjust:]
            adjust += amt
        s2 += ' ' * (j - i) + asetIndex.ljust(k - j)
        i = k
    long_lines = [s0, s1, s11, s2]
    outstr = '\n\n'.join(map('\n'.join, zip_longest(*mimic_wrap(long_lines), fillvalue=' '))).replace('~', ' ')
    outstr += '\n'
    if fAbbrevs:
        outstr += ' (' + ', '.join(('='.join(pair) for pair in fAbbrevs.items())) + ')'
        assert len(fAbbrevs) == len(dict(fAbbrevs)), 'Abbreviation clash'
    return outstr

def _annotation_ascii_FE_layer(overt, ni, feAbbrevs):
    if False:
        while True:
            i = 10
    'Helper for _annotation_ascii_FEs().'
    s1 = ''
    s2 = ''
    i = 0
    for (j, k, fename) in overt:
        s1 += ' ' * (j - i) + ('^' if fename.islower() else '-') * (k - j)
        short = fename[:k - j]
        if len(fename) > len(short):
            r = 0
            while short in feAbbrevs:
                if feAbbrevs[short] == fename:
                    break
                r += 1
                short = fename[:k - j - 1] + str(r)
            else:
                feAbbrevs[short] = fename
        s2 += ' ' * (j - i) + short.ljust(k - j)
        i = k
    sNI = ''
    if ni:
        sNI += ' [' + ', '.join((':'.join(x) for x in sorted(ni.items()))) + ']'
    return [s1, s2, sNI]

def _annotation_ascii_FEs(sent):
    if False:
        return 10
    "\n    ASCII string rendering of the sentence along with a single target and its FEs.\n    Secondary and tertiary FE layers are included if present.\n    'sent' can be an FE annotation set or an LU sentence with a single target.\n    Line-wrapped to limit the display width.\n    "
    feAbbrevs = OrderedDict()
    posspec = []
    posspec_separate = False
    for lyr in ('Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art'):
        if lyr in sent and sent[lyr]:
            for (a, b, lbl) in sent[lyr]:
                if lbl == 'X':
                    continue
                if any((1 for (x, y, felbl) in sent.FE[0] if x <= a < y or a <= x < b)):
                    posspec_separate = True
                posspec.append((a, b, lbl.lower().replace('-', '')))
    if posspec_separate:
        POSSPEC = _annotation_ascii_FE_layer(posspec, {}, feAbbrevs)
    FE1 = _annotation_ascii_FE_layer(sorted(sent.FE[0] + (posspec if not posspec_separate else [])), sent.FE[1], feAbbrevs)
    FE2 = FE3 = None
    if 'FE2' in sent:
        FE2 = _annotation_ascii_FE_layer(sent.FE2[0], sent.FE2[1], feAbbrevs)
        if 'FE3' in sent:
            FE3 = _annotation_ascii_FE_layer(sent.FE3[0], sent.FE3[1], feAbbrevs)
    for (i, j) in sent.Target:
        (FE1span, FE1name, FE1exp) = FE1
        if len(FE1span) < j:
            FE1span += ' ' * (j - len(FE1span))
        if len(FE1name) < j:
            FE1name += ' ' * (j - len(FE1name))
            FE1[1] = FE1name
        FE1[0] = FE1span[:i] + FE1span[i:j].replace(' ', '*').replace('-', '=') + FE1span[j:]
    long_lines = [sent.text]
    if posspec_separate:
        long_lines.extend(POSSPEC[:2])
    long_lines.extend([FE1[0], FE1[1] + FE1[2]])
    if FE2:
        long_lines.extend([FE2[0], FE2[1] + FE2[2]])
        if FE3:
            long_lines.extend([FE3[0], FE3[1] + FE3[2]])
    long_lines.append('')
    outstr = '\n'.join(map('\n'.join, zip_longest(*mimic_wrap(long_lines), fillvalue=' ')))
    if feAbbrevs:
        outstr += '(' + ', '.join(('='.join(pair) for pair in feAbbrevs.items())) + ')'
        assert len(feAbbrevs) == len(dict(feAbbrevs)), 'Abbreviation clash'
    outstr += '\n'
    return outstr

def _pretty_fe(fe):
    if False:
        while True:
            i = 10
    '\n    Helper function for pretty-printing a frame element.\n\n    :param fe: The frame element to be printed.\n    :type fe: AttrDict\n    :return: A nicely formatted string representation of the frame element.\n    :rtype: str\n    '
    fekeys = fe.keys()
    outstr = ''
    outstr += 'frame element ({0.ID}): {0.name}\n    of {1.name}({1.ID})\n'.format(fe, fe.frame)
    if 'definition' in fekeys:
        outstr += '[definition]\n'
        outstr += _pretty_longstring(fe.definition, '  ')
    if 'abbrev' in fekeys:
        outstr += f'[abbrev] {fe.abbrev}\n'
    if 'coreType' in fekeys:
        outstr += f'[coreType] {fe.coreType}\n'
    if 'requiresFE' in fekeys:
        outstr += '[requiresFE] '
        if fe.requiresFE is None:
            outstr += '<None>\n'
        else:
            outstr += f'{fe.requiresFE.name}({fe.requiresFE.ID})\n'
    if 'excludesFE' in fekeys:
        outstr += '[excludesFE] '
        if fe.excludesFE is None:
            outstr += '<None>\n'
        else:
            outstr += f'{fe.excludesFE.name}({fe.excludesFE.ID})\n'
    if 'semType' in fekeys:
        outstr += '[semType] '
        if fe.semType is None:
            outstr += '<None>\n'
        else:
            outstr += '\n  ' + f'{fe.semType.name}({fe.semType.ID})' + '\n'
    return outstr

def _pretty_frame(frame):
    if False:
        while True:
            i = 10
    '\n    Helper function for pretty-printing a frame.\n\n    :param frame: The frame to be printed.\n    :type frame: AttrDict\n    :return: A nicely formatted string representation of the frame.\n    :rtype: str\n    '
    outstr = ''
    outstr += 'frame ({0.ID}): {0.name}\n\n'.format(frame)
    outstr += f'[URL] {frame.URL}\n\n'
    outstr += '[definition]\n'
    outstr += _pretty_longstring(frame.definition, '  ') + '\n'
    outstr += f'[semTypes] {len(frame.semTypes)} semantic types\n'
    outstr += '  ' * (len(frame.semTypes) > 0) + ', '.join((f'{x.name}({x.ID})' for x in frame.semTypes)) + '\n' * (len(frame.semTypes) > 0)
    outstr += '\n[frameRelations] {} frame relations\n'.format(len(frame.frameRelations))
    outstr += '  ' + '\n  '.join((repr(frel) for frel in frame.frameRelations)) + '\n'
    outstr += f'\n[lexUnit] {len(frame.lexUnit)} lexical units\n'
    lustrs = []
    for (luName, lu) in sorted(frame.lexUnit.items()):
        tmpstr = f'{luName} ({lu.ID})'
        lustrs.append(tmpstr)
    outstr += '{}\n'.format(_pretty_longstring(', '.join(lustrs), prefix='  '))
    outstr += f'\n[FE] {len(frame.FE)} frame elements\n'
    fes = {}
    for (feName, fe) in sorted(frame.FE.items()):
        try:
            fes[fe.coreType].append(f'{feName} ({fe.ID})')
        except KeyError:
            fes[fe.coreType] = []
            fes[fe.coreType].append(f'{feName} ({fe.ID})')
    for ct in sorted(fes.keys(), key=lambda ct2: ['Core', 'Core-Unexpressed', 'Peripheral', 'Extra-Thematic'].index(ct2)):
        outstr += '{:>16}: {}\n'.format(ct, ', '.join(sorted(fes[ct])))
    outstr += '\n[FEcoreSets] {} frame element core sets\n'.format(len(frame.FEcoreSets))
    outstr += '  ' + '\n  '.join((', '.join([x.name for x in coreSet]) for coreSet in frame.FEcoreSets)) + '\n'
    return outstr

class FramenetError(Exception):
    """An exception class for framenet-related errors."""

class AttrDict(dict):
    """A class that wraps a dict and allows accessing the keys of the
    dict as if they were attributes. Taken from here:
    https://stackoverflow.com/a/14620633/8879

    >>> foo = {'a':1, 'b':2, 'c':3}
    >>> bar = AttrDict(foo)
    >>> pprint(dict(bar))
    {'a': 1, 'b': 2, 'c': 3}
    >>> bar.b
    2
    >>> bar.d = 4
    >>> pprint(dict(bar))
    {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        self[name] = value

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name == '_short_repr':
            return self._short_repr
        return self[name]

    def __getitem__(self, name):
        if False:
            print('Hello World!')
        v = super().__getitem__(name)
        if isinstance(v, Future):
            return v._data()
        return v

    def _short_repr(self):
        if False:
            return 10
        if '_type' in self:
            if self['_type'].endswith('relation'):
                return self.__repr__()
            try:
                return '<{} ID={} name={}>'.format(self['_type'], self['ID'], self['name'])
            except KeyError:
                try:
                    return '<{} name={}>'.format(self['_type'], self['name'])
                except KeyError:
                    return '<{} ID={}>'.format(self['_type'], self['ID'])
        else:
            return self.__repr__()

    def _str(self):
        if False:
            for i in range(10):
                print('nop')
        outstr = ''
        if '_type' not in self:
            outstr = _pretty_any(self)
        elif self['_type'] == 'frame':
            outstr = _pretty_frame(self)
        elif self['_type'] == 'fe':
            outstr = _pretty_fe(self)
        elif self['_type'] == 'lu':
            outstr = _pretty_lu(self)
        elif self['_type'] == 'luexemplars':
            outstr = _pretty_exemplars(self, self[0].LU)
        elif self['_type'] == 'fulltext_annotation':
            outstr = _pretty_fulltext_sentences(self)
        elif self['_type'] == 'lusentence':
            outstr = _pretty_annotation(self)
        elif self['_type'] == 'fulltext_sentence':
            outstr = _pretty_fulltext_sentence(self)
        elif self['_type'] in ('luannotationset', 'fulltext_annotationset'):
            outstr = _pretty_annotation(self, aset_level=True)
        elif self['_type'] == 'posannotationset':
            outstr = _pretty_pos(self)
        elif self['_type'] == 'semtype':
            outstr = _pretty_semtype(self)
        elif self['_type'] == 'framerelationtype':
            outstr = _pretty_frame_relation_type(self)
        elif self['_type'] == 'framerelation':
            outstr = _pretty_frame_relation(self)
        elif self['_type'] == 'ferelation':
            outstr = _pretty_fe_relation(self)
        else:
            outstr = _pretty_any(self)
        return outstr

    def __str__(self):
        if False:
            return 10
        return self._str()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.__str__()

class SpecialList(list):
    """
    A list subclass which adds a '_type' attribute for special printing
    (similar to an AttrDict, though this is NOT an AttrDict subclass).
    """

    def __init__(self, typ, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self._type = typ

    def _str(self):
        if False:
            print('Hello World!')
        outstr = ''
        assert self._type
        if len(self) == 0:
            outstr = '[]'
        elif self._type == 'luexemplars':
            outstr = _pretty_exemplars(self, self[0].LU)
        else:
            assert False, self._type
        return outstr

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self._str()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__str__()

class Future:
    """
    Wraps and acts as a proxy for a value to be loaded lazily (on demand).
    Adapted from https://gist.github.com/sergey-miryanov/2935416
    """

    def __init__(self, loader, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        :param loader: when called with no arguments, returns the value to be stored\n        :type loader: callable\n        '
        super().__init__(*args, **kwargs)
        self._loader = loader
        self._d = None

    def _data(self):
        if False:
            i = 10
            return i + 15
        if callable(self._loader):
            self._d = self._loader()
            self._loader = None
        return self._d

    def __nonzero__(self):
        if False:
            i = 10
            return i + 15
        return bool(self._data())

    def __len__(self):
        if False:
            return 10
        return len(self._data())

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        return self._data().__setitem__(key, value)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self._data().__getitem__(key)

    def __getattr__(self, key):
        if False:
            return 10
        return self._data().__getattr__(key)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self._data().__str__()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._data().__repr__()

class PrettyDict(AttrDict):
    """
    Displays an abbreviated repr of values where possible.
    Inherits from AttrDict, so a callable value will
    be lazily converted to an actual value.
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        _BREAK_LINES = kwargs.pop('breakLines', False)
        super().__init__(*args, **kwargs)
        dict.__setattr__(self, '_BREAK_LINES', _BREAK_LINES)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        parts = []
        for (k, v) in sorted(self.items()):
            kv = repr(k) + ': '
            try:
                kv += v._short_repr()
            except AttributeError:
                kv += repr(v)
            parts.append(kv)
        return '{' + (',\n ' if self._BREAK_LINES else ', ').join(parts) + '}'

class PrettyList(list):
    """
    Displays an abbreviated repr of only the first several elements, not the whole list.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._MAX_REPR_SIZE = kwargs.pop('maxReprSize', 60)
        self._BREAK_LINES = kwargs.pop('breakLines', False)
        super().__init__(*args, **kwargs)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        "\n        Return a string representation for this corpus view that is\n        similar to a list's representation; but if it would be more\n        than 60 characters long, it is truncated.\n        "
        pieces = []
        length = 5
        for elt in self:
            pieces.append(elt._short_repr())
            length += len(pieces[-1]) + 2
            if self._MAX_REPR_SIZE and length > self._MAX_REPR_SIZE and (len(pieces) > 2):
                return '[%s, ...]' % str(',\n ' if self._BREAK_LINES else ', ').join(pieces[:-1])
        return '[%s]' % str(',\n ' if self._BREAK_LINES else ', ').join(pieces)

class PrettyLazyMap(LazyMap):
    """
    Displays an abbreviated repr of only the first several elements, not the whole list.
    """
    _MAX_REPR_SIZE = 60

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a string representation for this corpus view that is\n        similar to a list's representation; but if it would be more\n        than 60 characters long, it is truncated.\n        "
        pieces = []
        length = 5
        for elt in self:
            pieces.append(elt._short_repr())
            length += len(pieces[-1]) + 2
            if length > self._MAX_REPR_SIZE and len(pieces) > 2:
                return '[%s, ...]' % ', '.join(pieces[:-1])
        return '[%s]' % ', '.join(pieces)

class PrettyLazyIteratorList(LazyIteratorList):
    """
    Displays an abbreviated repr of only the first several elements, not the whole list.
    """
    _MAX_REPR_SIZE = 60

    def __repr__(self):
        if False:
            return 10
        "\n        Return a string representation for this corpus view that is\n        similar to a list's representation; but if it would be more\n        than 60 characters long, it is truncated.\n        "
        pieces = []
        length = 5
        for elt in self:
            pieces.append(elt._short_repr())
            length += len(pieces[-1]) + 2
            if length > self._MAX_REPR_SIZE and len(pieces) > 2:
                return '[%s, ...]' % ', '.join(pieces[:-1])
        return '[%s]' % ', '.join(pieces)

class PrettyLazyConcatenation(LazyConcatenation):
    """
    Displays an abbreviated repr of only the first several elements, not the whole list.
    """
    _MAX_REPR_SIZE = 60

    def __repr__(self):
        if False:
            return 10
        "\n        Return a string representation for this corpus view that is\n        similar to a list's representation; but if it would be more\n        than 60 characters long, it is truncated.\n        "
        pieces = []
        length = 5
        for elt in self:
            pieces.append(elt._short_repr())
            length += len(pieces[-1]) + 2
            if length > self._MAX_REPR_SIZE and len(pieces) > 2:
                return '[%s, ...]' % ', '.join(pieces[:-1])
        return '[%s]' % ', '.join(pieces)

    def __add__(self, other):
        if False:
            while True:
                i = 10
        'Return a list concatenating self with other.'
        return PrettyLazyIteratorList(itertools.chain(self, other))

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        'Return a list concatenating other with self.'
        return PrettyLazyIteratorList(itertools.chain(other, self))

class FramenetCorpusReader(XMLCorpusReader):
    """A corpus reader for the Framenet Corpus.

    >>> from nltk.corpus import framenet as fn
    >>> fn.lu(3238).frame.lexUnit['glint.v'] is fn.lu(3238)
    True
    >>> fn.frame_by_name('Replacing') is fn.lus('replace.v')[0].frame
    True
    >>> fn.lus('prejudice.n')[0].frame.frameRelations == fn.frame_relations('Partiality')
    True
    """
    _bad_statuses = ['Problem']
    "\n    When loading LUs for a frame, those whose status is in this list will be ignored.\n    Due to caching, if user code modifies this, it should do so before loading any data.\n    'Problem' should always be listed for FrameNet 1.5, as these LUs are not included\n    in the XML index.\n    "
    _warnings = False

    def warnings(self, v):
        if False:
            print('Hello World!')
        'Enable or disable warnings of data integrity issues as they are encountered.\n        If v is truthy, warnings will be enabled.\n\n        (This is a function rather than just an attribute/property to ensure that if\n        enabling warnings is the first action taken, the corpus reader is instantiated first.)\n        '
        self._warnings = v

    def __init__(self, root, fileids):
        if False:
            print('Hello World!')
        XMLCorpusReader.__init__(self, root, fileids)
        self._frame_dir = 'frame'
        self._lu_dir = 'lu'
        self._fulltext_dir = 'fulltext'
        self._fnweb_url = 'https://framenet2.icsi.berkeley.edu/fnReports/data'
        self._frame_idx = None
        self._cached_frames = {}
        self._lu_idx = None
        self._fulltext_idx = None
        self._semtypes = None
        self._freltyp_idx = None
        self._frel_idx = None
        self._ferel_idx = None
        self._frel_f_idx = None
        self._readme = 'README.txt'

    def help(self, attrname=None):
        if False:
            i = 10
            return i + 15
        'Display help information summarizing the main methods.'
        if attrname is not None:
            return help(self.__getattribute__(attrname))
        msg = '\nCitation: Nathan Schneider and Chuck Wooters (2017),\n"The NLTK FrameNet API: Designing for Discoverability with a Rich Linguistic Resource".\nProceedings of EMNLP: System Demonstrations. https://arxiv.org/abs/1703.07438\n\nUse the following methods to access data in FrameNet.\nProvide a method name to `help()` for more information.\n\nFRAMES\n======\n\nframe() to look up a frame by its exact name or ID\nframes() to get frames matching a name pattern\nframes_by_lemma() to get frames containing an LU matching a name pattern\nframe_ids_and_names() to get a mapping from frame IDs to names\n\nFRAME ELEMENTS\n==============\n\nfes() to get frame elements (a.k.a. roles) matching a name pattern, optionally constrained\n  by a frame name pattern\n\nLEXICAL UNITS\n=============\n\nlu() to look up an LU by its ID\nlus() to get lexical units matching a name pattern, optionally constrained by frame\nlu_ids_and_names() to get a mapping from LU IDs to names\n\nRELATIONS\n=========\n\nframe_relation_types() to get the different kinds of frame-to-frame relations\n  (Inheritance, Subframe, Using, etc.).\nframe_relations() to get the relation instances, optionally constrained by\n  frame(s) or relation type\nfe_relations() to get the frame element pairs belonging to a frame-to-frame relation\n\nSEMANTIC TYPES\n==============\n\nsemtypes() to get the different kinds of semantic types that can be applied to\n  FEs, LUs, and entire frames\nsemtype() to look up a particular semtype by name, ID, or abbreviation\nsemtype_inherits() to check whether two semantic types have a subtype-supertype\n  relationship in the semtype hierarchy\npropagate_semtypes() to apply inference rules that distribute semtypes over relations\n  between FEs\n\nANNOTATIONS\n===========\n\nannotations() to get annotation sets, in which a token in a sentence is annotated\n  with a lexical unit in a frame, along with its frame elements and their syntactic properties;\n  can be constrained by LU name pattern and limited to lexicographic exemplars or full-text.\n  Sentences of full-text annotation can have multiple annotation sets.\nsents() to get annotated sentences illustrating one or more lexical units\nexemplars() to get sentences of lexicographic annotation, most of which have\n  just 1 annotation set; can be constrained by LU name pattern, frame, and overt FE(s)\ndoc() to look up a document of full-text annotation by its ID\ndocs() to get documents of full-text annotation that match a name pattern\ndocs_metadata() to get metadata about all full-text documents without loading them\nft_sents() to iterate over sentences of full-text annotation\n\nUTILITIES\n=========\n\nbuildindexes() loads metadata about all frames, LUs, etc. into memory to avoid\n  delay when one is accessed for the first time. It does not load annotations.\nreadme() gives the text of the FrameNet README file\nwarnings(True) to display corpus consistency warnings when loading data\n        '
        print(msg)

    def _buildframeindex(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._frel_idx:
            self._buildrelationindex()
        self._frame_idx = {}
        with XMLCorpusView(self.abspath('frameIndex.xml'), 'frameIndex/frame', self._handle_elt) as view:
            for f in view:
                self._frame_idx[f['ID']] = f

    def _buildcorpusindex(self):
        if False:
            print('Hello World!')
        self._fulltext_idx = {}
        with XMLCorpusView(self.abspath('fulltextIndex.xml'), 'fulltextIndex/corpus', self._handle_fulltextindex_elt) as view:
            for doclist in view:
                for doc in doclist:
                    self._fulltext_idx[doc.ID] = doc

    def _buildluindex(self):
        if False:
            i = 10
            return i + 15
        self._lu_idx = {}
        with XMLCorpusView(self.abspath('luIndex.xml'), 'luIndex/lu', self._handle_elt) as view:
            for lu in view:
                self._lu_idx[lu['ID']] = lu

    def _buildrelationindex(self):
        if False:
            return 10
        self._freltyp_idx = {}
        self._frel_idx = {}
        self._frel_f_idx = defaultdict(set)
        self._ferel_idx = {}
        with XMLCorpusView(self.abspath('frRelation.xml'), 'frameRelations/frameRelationType', self._handle_framerelationtype_elt) as view:
            for freltyp in view:
                self._freltyp_idx[freltyp.ID] = freltyp
                for frel in freltyp.frameRelations:
                    supF = frel.superFrame = frel[freltyp.superFrameName] = Future((lambda fID: lambda : self.frame_by_id(fID))(frel.supID))
                    subF = frel.subFrame = frel[freltyp.subFrameName] = Future((lambda fID: lambda : self.frame_by_id(fID))(frel.subID))
                    self._frel_idx[frel.ID] = frel
                    self._frel_f_idx[frel.supID].add(frel.ID)
                    self._frel_f_idx[frel.subID].add(frel.ID)
                    for ferel in frel.feRelations:
                        ferel.superFrame = supF
                        ferel.subFrame = subF
                        ferel.superFE = Future((lambda fer: lambda : fer.superFrame.FE[fer.superFEName])(ferel))
                        ferel.subFE = Future((lambda fer: lambda : fer.subFrame.FE[fer.subFEName])(ferel))
                        self._ferel_idx[ferel.ID] = ferel

    def _warn(self, *message, **kwargs):
        if False:
            return 10
        if self._warnings:
            kwargs.setdefault('file', sys.stderr)
            print(*message, **kwargs)

    def buildindexes(self):
        if False:
            while True:
                i = 10
        '\n        Build the internal indexes to make look-ups faster.\n        '
        self._buildframeindex()
        self._buildluindex()
        self._buildcorpusindex()
        self._buildrelationindex()

    def doc(self, fn_docid):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the annotated document whose id number is\n        ``fn_docid``. This id number can be obtained by calling the\n        Documents() function.\n\n        The dict that is returned from this function will contain the\n        following keys:\n\n        - '_type'      : 'fulltextannotation'\n        - 'sentence'   : a list of sentences in the document\n           - Each item in the list is a dict containing the following keys:\n              - 'ID'    : the ID number of the sentence\n              - '_type' : 'sentence'\n              - 'text'  : the text of the sentence\n              - 'paragNo' : the paragraph number\n              - 'sentNo'  : the sentence number\n              - 'docID'   : the document ID number\n              - 'corpID'  : the corpus ID number\n              - 'aPos'    : the annotation position\n              - 'annotationSet' : a list of annotation layers for the sentence\n                 - Each item in the list is a dict containing the following keys:\n                    - 'ID'       : the ID number of the annotation set\n                    - '_type'    : 'annotationset'\n                    - 'status'   : either 'MANUAL' or 'UNANN'\n                    - 'luName'   : (only if status is 'MANUAL')\n                    - 'luID'     : (only if status is 'MANUAL')\n                    - 'frameID'  : (only if status is 'MANUAL')\n                    - 'frameName': (only if status is 'MANUAL')\n                    - 'layer' : a list of labels for the layer\n                       - Each item in the layer is a dict containing the following keys:\n                          - '_type': 'layer'\n                          - 'rank'\n                          - 'name'\n                          - 'label' : a list of labels in the layer\n                             - Each item is a dict containing the following keys:\n                                - 'start'\n                                - 'end'\n                                - 'name'\n                                - 'feID' (optional)\n\n        :param fn_docid: The Framenet id number of the document\n        :type fn_docid: int\n        :return: Information about the annotated document\n        :rtype: dict\n        "
        try:
            xmlfname = self._fulltext_idx[fn_docid].filename
        except TypeError:
            self._buildcorpusindex()
            xmlfname = self._fulltext_idx[fn_docid].filename
        except KeyError as e:
            raise FramenetError(f'Unknown document id: {fn_docid}') from e
        locpath = os.path.join(f'{self._root}', self._fulltext_dir, xmlfname)
        with XMLCorpusView(locpath, 'fullTextAnnotation') as view:
            elt = view[0]
        info = self._handle_fulltextannotation_elt(elt)
        for (k, v) in self._fulltext_idx[fn_docid].items():
            info[k] = v
        return info

    def frame_by_id(self, fn_fid, ignorekeys=[]):
        if False:
            while True:
                i = 10
        '\n        Get the details for the specified Frame using the frame\'s id\n        number.\n\n        Usage examples:\n\n        >>> from nltk.corpus import framenet as fn\n        >>> f = fn.frame_by_id(256)\n        >>> f.ID\n        256\n        >>> f.name\n        \'Medical_specialties\'\n        >>> f.definition # doctest: +NORMALIZE_WHITESPACE\n        "This frame includes words that name medical specialties and is closely related to the\n        Medical_professionals frame.  The FE Type characterizing a sub-are in a Specialty may also be\n        expressed. \'Ralph practices paediatric oncology.\'"\n\n        :param fn_fid: The Framenet id number of the frame\n        :type fn_fid: int\n        :param ignorekeys: The keys to ignore. These keys will not be\n            included in the output. (optional)\n        :type ignorekeys: list(str)\n        :return: Information about a frame\n        :rtype: dict\n\n        Also see the ``frame()`` function for details about what is\n        contained in the dict that is returned.\n        '
        try:
            fentry = self._frame_idx[fn_fid]
            if '_type' in fentry:
                return fentry
            name = fentry['name']
        except TypeError:
            self._buildframeindex()
            name = self._frame_idx[fn_fid]['name']
        except KeyError as e:
            raise FramenetError(f'Unknown frame id: {fn_fid}') from e
        return self.frame_by_name(name, ignorekeys, check_cache=False)

    def frame_by_name(self, fn_fname, ignorekeys=[], check_cache=True):
        if False:
            i = 10
            return i + 15
        '\n        Get the details for the specified Frame using the frame\'s name.\n\n        Usage examples:\n\n        >>> from nltk.corpus import framenet as fn\n        >>> f = fn.frame_by_name(\'Medical_specialties\')\n        >>> f.ID\n        256\n        >>> f.name\n        \'Medical_specialties\'\n        >>> f.definition # doctest: +NORMALIZE_WHITESPACE\n         "This frame includes words that name medical specialties and is closely related to the\n          Medical_professionals frame.  The FE Type characterizing a sub-are in a Specialty may also be\n          expressed. \'Ralph practices paediatric oncology.\'"\n\n        :param fn_fname: The name of the frame\n        :type fn_fname: str\n        :param ignorekeys: The keys to ignore. These keys will not be\n            included in the output. (optional)\n        :type ignorekeys: list(str)\n        :return: Information about a frame\n        :rtype: dict\n\n        Also see the ``frame()`` function for details about what is\n        contained in the dict that is returned.\n        '
        if check_cache and fn_fname in self._cached_frames:
            return self._frame_idx[self._cached_frames[fn_fname]]
        elif not self._frame_idx:
            self._buildframeindex()
        locpath = os.path.join(f'{self._root}', self._frame_dir, fn_fname + '.xml')
        try:
            with XMLCorpusView(locpath, 'frame') as view:
                elt = view[0]
        except OSError as e:
            raise FramenetError(f'Unknown frame: {fn_fname}') from e
        fentry = self._handle_frame_elt(elt, ignorekeys)
        assert fentry
        fentry.URL = self._fnweb_url + '/' + self._frame_dir + '/' + fn_fname + '.xml'
        for st in fentry.semTypes:
            if st.rootType.name == 'Lexical_type':
                for lu in fentry.lexUnit.values():
                    if not any((x is st for x in lu.semTypes)):
                        lu.semTypes.append(st)
        self._frame_idx[fentry.ID] = fentry
        self._cached_frames[fentry.name] = fentry.ID
        '\n        # now set up callables to resolve the LU pointers lazily.\n        # (could also do this here--caching avoids infinite recursion.)\n        for luName,luinfo in fentry.lexUnit.items():\n            fentry.lexUnit[luName] = (lambda luID: Future(lambda: self.lu(luID)))(luinfo.ID)\n        '
        return fentry

    def frame(self, fn_fid_or_fname, ignorekeys=[]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the details for the specified Frame using the frame\'s name\n        or id number.\n\n        Usage examples:\n\n        >>> from nltk.corpus import framenet as fn\n        >>> f = fn.frame(256)\n        >>> f.name\n        \'Medical_specialties\'\n        >>> f = fn.frame(\'Medical_specialties\')\n        >>> f.ID\n        256\n        >>> # ensure non-ASCII character in definition doesn\'t trigger an encoding error:\n        >>> fn.frame(\'Imposing_obligation\') # doctest: +ELLIPSIS\n        frame (1494): Imposing_obligation...\n\n\n        The dict that is returned from this function will contain the\n        following information about the Frame:\n\n        - \'name\'       : the name of the Frame (e.g. \'Birth\', \'Apply_heat\', etc.)\n        - \'definition\' : textual definition of the Frame\n        - \'ID\'         : the internal ID number of the Frame\n        - \'semTypes\'   : a list of semantic types for this frame\n           - Each item in the list is a dict containing the following keys:\n              - \'name\' : can be used with the semtype() function\n              - \'ID\'   : can be used with the semtype() function\n\n        - \'lexUnit\'    : a dict containing all of the LUs for this frame.\n                         The keys in this dict are the names of the LUs and\n                         the value for each key is itself a dict containing\n                         info about the LU (see the lu() function for more info.)\n\n        - \'FE\' : a dict containing the Frame Elements that are part of this frame\n                 The keys in this dict are the names of the FEs (e.g. \'Body_system\')\n                 and the values are dicts containing the following keys\n\n              - \'definition\' : The definition of the FE\n              - \'name\'       : The name of the FE e.g. \'Body_system\'\n              - \'ID\'         : The id number\n              - \'_type\'      : \'fe\'\n              - \'abbrev\'     : Abbreviation e.g. \'bod\'\n              - \'coreType\'   : one of "Core", "Peripheral", or "Extra-Thematic"\n              - \'semType\'    : if not None, a dict with the following two keys:\n                 - \'name\' : name of the semantic type. can be used with\n                            the semtype() function\n                 - \'ID\'   : id number of the semantic type. can be used with\n                            the semtype() function\n              - \'requiresFE\' : if not None, a dict with the following two keys:\n                 - \'name\' : the name of another FE in this frame\n                 - \'ID\'   : the id of the other FE in this frame\n              - \'excludesFE\' : if not None, a dict with the following two keys:\n                 - \'name\' : the name of another FE in this frame\n                 - \'ID\'   : the id of the other FE in this frame\n\n        - \'frameRelation\'      : a list of objects describing frame relations\n        - \'FEcoreSets\'  : a list of Frame Element core sets for this frame\n           - Each item in the list is a list of FE objects\n\n        :param fn_fid_or_fname: The Framenet name or id number of the frame\n        :type fn_fid_or_fname: int or str\n        :param ignorekeys: The keys to ignore. These keys will not be\n            included in the output. (optional)\n        :type ignorekeys: list(str)\n        :return: Information about a frame\n        :rtype: dict\n        '
        if isinstance(fn_fid_or_fname, str):
            f = self.frame_by_name(fn_fid_or_fname, ignorekeys)
        else:
            f = self.frame_by_id(fn_fid_or_fname, ignorekeys)
        return f

    def frames_by_lemma(self, pat):
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of all frames that contain LUs in which the\n        ``name`` attribute of the LU matches the given regular expression\n        ``pat``. Note that LU names are composed of "lemma.POS", where\n        the "lemma" part can be made up of either a single lexeme\n        (e.g. \'run\') or multiple lexemes (e.g. \'a little\').\n\n        Note: if you are going to be doing a lot of this type of\n        searching, you\'d want to build an index that maps from lemmas to\n        frames because each time frames_by_lemma() is called, it has to\n        search through ALL of the frame XML files in the db.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> from nltk.corpus.reader.framenet import PrettyList\n        >>> PrettyList(sorted(fn.frames_by_lemma(r\'(?i)a little\'), key=itemgetter(\'ID\'))) # doctest: +ELLIPSIS\n        [<frame ID=189 name=Quanti...>, <frame ID=2001 name=Degree>]\n\n        :return: A list of frame objects.\n        :rtype: list(AttrDict)\n        '
        return PrettyList((f for f in self.frames() if any((re.search(pat, luName) for luName in f.lexUnit))))

    def lu_basic(self, fn_luid):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns basic information about the LU whose id is\n        ``fn_luid``. This is basically just a wrapper around the\n        ``lu()`` function with "subCorpus" info excluded.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> lu = PrettyDict(fn.lu_basic(256), breakLines=True)\n        >>> # ellipses account for differences between FN 1.5 and 1.7\n        >>> lu # doctest: +ELLIPSIS\n        {\'ID\': 256,\n         \'POS\': \'V\',\n         \'URL\': \'https://framenet2.icsi.berkeley.edu/fnReports/data/lu/lu256.xml\',\n         \'_type\': \'lu\',\n         \'cBy\': ...,\n         \'cDate\': \'02/08/2001 01:27:50 PST Thu\',\n         \'definition\': \'COD: be aware of beforehand; predict.\',\n         \'definitionMarkup\': \'COD: be aware of beforehand; predict.\',\n         \'frame\': <frame ID=26 name=Expectation>,\n         \'lemmaID\': 15082,\n         \'lexemes\': [{\'POS\': \'V\', \'breakBefore\': \'false\', \'headword\': \'false\', \'name\': \'foresee\', \'order\': 1}],\n         \'name\': \'foresee.v\',\n         \'semTypes\': [],\n         \'sentenceCount\': {\'annotated\': ..., \'total\': ...},\n         \'status\': \'FN1_Sent\'}\n\n        :param fn_luid: The id number of the desired LU\n        :type fn_luid: int\n        :return: Basic information about the lexical unit\n        :rtype: dict\n        '
        return self.lu(fn_luid, ignorekeys=['subCorpus', 'exemplars'])

    def lu(self, fn_luid, ignorekeys=[], luName=None, frameID=None, frameName=None):
        if False:
            i = 10
            return i + 15
        '\n        Access a lexical unit by its ID. luName, frameID, and frameName are used\n        only in the event that the LU does not have a file in the database\n        (which is the case for LUs with "Problem" status); in this case,\n        a placeholder LU is created which just contains its name, ID, and frame.\n\n\n        Usage examples:\n\n        >>> from nltk.corpus import framenet as fn\n        >>> fn.lu(256).name\n        \'foresee.v\'\n        >>> fn.lu(256).definition\n        \'COD: be aware of beforehand; predict.\'\n        >>> fn.lu(256).frame.name\n        \'Expectation\'\n        >>> list(map(PrettyDict, fn.lu(256).lexemes))\n        [{\'POS\': \'V\', \'breakBefore\': \'false\', \'headword\': \'false\', \'name\': \'foresee\', \'order\': 1}]\n\n        >>> fn.lu(227).exemplars[23] # doctest: +NORMALIZE_WHITESPACE\n        exemplar sentence (352962):\n        [sentNo] 0\n        [aPos] 59699508\n        <BLANKLINE>\n        [LU] (227) guess.v in Coming_to_believe\n        <BLANKLINE>\n        [frame] (23) Coming_to_believe\n        <BLANKLINE>\n        [annotationSet] 2 annotation sets\n        <BLANKLINE>\n        [POS] 18 tags\n        <BLANKLINE>\n        [POS_tagset] BNC\n        <BLANKLINE>\n        [GF] 3 relations\n        <BLANKLINE>\n        [PT] 3 phrases\n        <BLANKLINE>\n        [Other] 1 entry\n        <BLANKLINE>\n        [text] + [Target] + [FE]\n        <BLANKLINE>\n        When he was inside the house , Culley noticed the characteristic\n                                                      ------------------\n                                                      Content\n        <BLANKLINE>\n        he would n\'t have guessed at .\n        --                ******* --\n        Co                        C1 [Evidence:INI]\n         (Co=Cognizer, C1=Content)\n        <BLANKLINE>\n        <BLANKLINE>\n\n        The dict that is returned from this function will contain most of the\n        following information about the LU. Note that some LUs do not contain\n        all of these pieces of information - particularly \'totalAnnotated\' and\n        \'incorporatedFE\' may be missing in some LUs:\n\n        - \'name\'       : the name of the LU (e.g. \'merger.n\')\n        - \'definition\' : textual definition of the LU\n        - \'ID\'         : the internal ID number of the LU\n        - \'_type\'      : \'lu\'\n        - \'status\'     : e.g. \'Created\'\n        - \'frame\'      : Frame that this LU belongs to\n        - \'POS\'        : the part of speech of this LU (e.g. \'N\')\n        - \'totalAnnotated\' : total number of examples annotated with this LU\n        - \'incorporatedFE\' : FE that incorporates this LU (e.g. \'Ailment\')\n        - \'sentenceCount\'  : a dict with the following two keys:\n                 - \'annotated\': number of sentences annotated with this LU\n                 - \'total\'    : total number of sentences with this LU\n\n        - \'lexemes\'  : a list of dicts describing the lemma of this LU.\n           Each dict in the list contains these keys:\n\n           - \'POS\'     : part of speech e.g. \'N\'\n           - \'name\'    : either single-lexeme e.g. \'merger\' or\n                         multi-lexeme e.g. \'a little\'\n           - \'order\': the order of the lexeme in the lemma (starting from 1)\n           - \'headword\': a boolean (\'true\' or \'false\')\n           - \'breakBefore\': Can this lexeme be separated from the previous lexeme?\n                Consider: "take over.v" as in::\n\n                         Germany took over the Netherlands in 2 days.\n                         Germany took the Netherlands over in 2 days.\n\n                In this case, \'breakBefore\' would be "true" for the lexeme\n                "over". Contrast this with "take after.v" as in::\n\n                         Mary takes after her grandmother.\n                        *Mary takes her grandmother after.\n\n                In this case, \'breakBefore\' would be "false" for the lexeme "after"\n\n        - \'lemmaID\'    : Can be used to connect lemmas in different LUs\n        - \'semTypes\'   : a list of semantic type objects for this LU\n        - \'subCorpus\'  : a list of subcorpora\n           - Each item in the list is a dict containing the following keys:\n              - \'name\' :\n              - \'sentence\' : a list of sentences in the subcorpus\n                 - each item in the list is a dict with the following keys:\n                    - \'ID\':\n                    - \'sentNo\':\n                    - \'text\': the text of the sentence\n                    - \'aPos\':\n                    - \'annotationSet\': a list of annotation sets\n                       - each item in the list is a dict with the following keys:\n                          - \'ID\':\n                          - \'status\':\n                          - \'layer\': a list of layers\n                             - each layer is a dict containing the following keys:\n                                - \'name\': layer name (e.g. \'BNC\')\n                                - \'rank\':\n                                - \'label\': a list of labels for the layer\n                                   - each label is a dict containing the following keys:\n                                      - \'start\': start pos of label in sentence \'text\' (0-based)\n                                      - \'end\': end pos of label in sentence \'text\' (0-based)\n                                      - \'name\': name of label (e.g. \'NN1\')\n\n        Under the hood, this implementation looks up the lexical unit information\n        in the *frame* definition file. That file does not contain\n        corpus annotations, so the LU files will be accessed on demand if those are\n        needed. In principle, valence patterns could be loaded here too,\n        though these are not currently supported.\n\n        :param fn_luid: The id number of the lexical unit\n        :type fn_luid: int\n        :param ignorekeys: The keys to ignore. These keys will not be\n            included in the output. (optional)\n        :type ignorekeys: list(str)\n        :return: All information about the lexical unit\n        :rtype: dict\n        '
        if not self._lu_idx:
            self._buildluindex()
        OOV = object()
        luinfo = self._lu_idx.get(fn_luid, OOV)
        if luinfo is OOV:
            self._warn('LU ID not found: {} ({}) in {} ({})'.format(luName, fn_luid, frameName, frameID))
            luinfo = AttrDict({'_type': 'lu', 'ID': fn_luid, 'name': luName, 'frameID': frameID, 'status': 'Problem'})
            f = self.frame_by_id(luinfo.frameID)
            assert f.name == frameName, (f.name, frameName)
            luinfo['frame'] = f
            self._lu_idx[fn_luid] = luinfo
        elif '_type' not in luinfo:
            f = self.frame_by_id(luinfo.frameID)
            luinfo = self._lu_idx[fn_luid]
        if ignorekeys:
            return AttrDict({k: v for (k, v) in luinfo.items() if k not in ignorekeys})
        return luinfo

    def _lu_file(self, lu, ignorekeys=[]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Augment the LU information that was loaded from the frame file\n        with additional information from the LU file.\n        '
        fn_luid = lu.ID
        fname = f'lu{fn_luid}.xml'
        locpath = os.path.join(f'{self._root}', self._lu_dir, fname)
        if not self._lu_idx:
            self._buildluindex()
        try:
            with XMLCorpusView(locpath, 'lexUnit') as view:
                elt = view[0]
        except OSError as e:
            raise FramenetError(f'Unknown LU id: {fn_luid}') from e
        lu2 = self._handle_lexunit_elt(elt, ignorekeys)
        lu.URL = self._fnweb_url + '/' + self._lu_dir + '/' + fname
        lu.subCorpus = lu2.subCorpus
        lu.exemplars = SpecialList('luexemplars', [sent for subc in lu.subCorpus for sent in subc.sentence])
        for sent in lu.exemplars:
            sent['LU'] = lu
            sent['frame'] = lu.frame
            for aset in sent.annotationSet:
                aset['LU'] = lu
                aset['frame'] = lu.frame
        return lu

    def _loadsemtypes(self):
        if False:
            print('Hello World!')
        'Create the semantic types index.'
        self._semtypes = AttrDict()
        with XMLCorpusView(self.abspath('semTypes.xml'), 'semTypes/semType', self._handle_semtype_elt) as view:
            for st in view:
                n = st['name']
                a = st['abbrev']
                i = st['ID']
                self._semtypes[n] = i
                self._semtypes[a] = i
                self._semtypes[i] = st
        roots = []
        for st in self.semtypes():
            if st.superType:
                st.superType = self.semtype(st.superType.supID)
                st.superType.subTypes.append(st)
            else:
                if st not in roots:
                    roots.append(st)
                st.rootType = st
        queue = list(roots)
        assert queue
        while queue:
            st = queue.pop(0)
            for child in st.subTypes:
                child.rootType = st.rootType
                queue.append(child)

    def propagate_semtypes(self):
        if False:
            i = 10
            return i + 15
        '\n        Apply inference rules to distribute semtypes over relations between FEs.\n        For FrameNet 1.5, this results in 1011 semtypes being propagated.\n        (Not done by default because it requires loading all frame files,\n        which takes several seconds. If this needed to be fast, it could be rewritten\n        to traverse the neighboring relations on demand for each FE semtype.)\n\n        >>> from nltk.corpus import framenet as fn\n        >>> x = sum(1 for f in fn.frames() for fe in f.FE.values() if fe.semType)\n        >>> fn.propagate_semtypes()\n        >>> y = sum(1 for f in fn.frames() for fe in f.FE.values() if fe.semType)\n        >>> y-x > 1000\n        True\n        '
        if not self._semtypes:
            self._loadsemtypes()
        if not self._ferel_idx:
            self._buildrelationindex()
        changed = True
        i = 0
        nPropagations = 0
        while changed:
            i += 1
            changed = False
            for ferel in self.fe_relations():
                superST = ferel.superFE.semType
                subST = ferel.subFE.semType
                try:
                    if superST and superST is not subST:
                        assert subST is None or self.semtype_inherits(subST, superST), (superST.name, ferel, subST.name)
                        if subST is None:
                            ferel.subFE.semType = subST = superST
                            changed = True
                            nPropagations += 1
                    if ferel.type.name in ['Perspective_on', 'Subframe', 'Precedes'] and subST and (subST is not superST):
                        assert superST is None, (superST.name, ferel, subST.name)
                        ferel.superFE.semType = superST = subST
                        changed = True
                        nPropagations += 1
                except AssertionError as ex:
                    continue

    def semtype(self, key):
        if False:
            return 10
        "\n        >>> from nltk.corpus import framenet as fn\n        >>> fn.semtype(233).name\n        'Temperature'\n        >>> fn.semtype(233).abbrev\n        'Temp'\n        >>> fn.semtype('Temperature').ID\n        233\n\n        :param key: The name, abbreviation, or id number of the semantic type\n        :type key: string or int\n        :return: Information about a semantic type\n        :rtype: dict\n        "
        if isinstance(key, int):
            stid = key
        else:
            try:
                stid = self._semtypes[key]
            except TypeError:
                self._loadsemtypes()
                stid = self._semtypes[key]
        try:
            st = self._semtypes[stid]
        except TypeError:
            self._loadsemtypes()
            st = self._semtypes[stid]
        return st

    def semtype_inherits(self, st, superST):
        if False:
            while True:
                i = 10
        if not isinstance(st, dict):
            st = self.semtype(st)
        if not isinstance(superST, dict):
            superST = self.semtype(superST)
        par = st.superType
        while par:
            if par is superST:
                return True
            par = par.superType
        return False

    def frames(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Obtain details for a specific frame.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> len(fn.frames()) in (1019, 1221)    # FN 1.5 and 1.7, resp.\n        True\n        >>> x = PrettyList(fn.frames(r\'(?i)crim\'), maxReprSize=0, breakLines=True)\n        >>> x.sort(key=itemgetter(\'ID\'))\n        >>> x\n        [<frame ID=200 name=Criminal_process>,\n         <frame ID=500 name=Criminal_investigation>,\n         <frame ID=692 name=Crime_scenario>,\n         <frame ID=700 name=Committing_crime>]\n\n        A brief intro to Frames (excerpted from "FrameNet II: Extended\n        Theory and Practice" by Ruppenhofer et. al., 2010):\n\n        A Frame is a script-like conceptual structure that describes a\n        particular type of situation, object, or event along with the\n        participants and props that are needed for that Frame. For\n        example, the "Apply_heat" frame describes a common situation\n        involving a Cook, some Food, and a Heating_Instrument, and is\n        evoked by words such as bake, blanch, boil, broil, brown,\n        simmer, steam, etc.\n\n        We call the roles of a Frame "frame elements" (FEs) and the\n        frame-evoking words are called "lexical units" (LUs).\n\n        FrameNet includes relations between Frames. Several types of\n        relations are defined, of which the most important are:\n\n           - Inheritance: An IS-A relation. The child frame is a subtype\n             of the parent frame, and each FE in the parent is bound to\n             a corresponding FE in the child. An example is the\n             "Revenge" frame which inherits from the\n             "Rewards_and_punishments" frame.\n\n           - Using: The child frame presupposes the parent frame as\n             background, e.g the "Speed" frame "uses" (or presupposes)\n             the "Motion" frame; however, not all parent FEs need to be\n             bound to child FEs.\n\n           - Subframe: The child frame is a subevent of a complex event\n             represented by the parent, e.g. the "Criminal_process" frame\n             has subframes of "Arrest", "Arraignment", "Trial", and\n             "Sentencing".\n\n           - Perspective_on: The child frame provides a particular\n             perspective on an un-perspectivized parent frame. A pair of\n             examples consists of the "Hiring" and "Get_a_job" frames,\n             which perspectivize the "Employment_start" frame from the\n             Employer\'s and the Employee\'s point of view, respectively.\n\n        :param name: A regular expression pattern used to match against\n            Frame names. If \'name\' is None, then a list of all\n            Framenet Frames will be returned.\n        :type name: str\n        :return: A list of matching Frames (or all Frames).\n        :rtype: list(AttrDict)\n        '
        try:
            fIDs = list(self._frame_idx.keys())
        except AttributeError:
            self._buildframeindex()
            fIDs = list(self._frame_idx.keys())
        if name is not None:
            return PrettyList((self.frame(fID) for (fID, finfo) in self.frame_ids_and_names(name).items()))
        else:
            return PrettyLazyMap(self.frame, fIDs)

    def frame_ids_and_names(self, name=None):
        if False:
            i = 10
            return i + 15
        '\n        Uses the frame index, which is much faster than looking up each frame definition\n        if only the names and IDs are needed.\n        '
        if not self._frame_idx:
            self._buildframeindex()
        return {fID: finfo.name for (fID, finfo) in self._frame_idx.items() if name is None or re.search(name, finfo.name) is not None}

    def fes(self, name=None, frame=None):
        if False:
            print('Hello World!')
        "\n        Lists frame element objects. If 'name' is provided, this is treated as\n        a case-insensitive regular expression to filter by frame name.\n        (Case-insensitivity is because casing of frame element names is not always\n        consistent across frames.) Specify 'frame' to filter by a frame name pattern,\n        ID, or object.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> fn.fes('Noise_maker')\n        [<fe ID=6043 name=Noise_maker>]\n        >>> sorted([(fe.frame.name,fe.name) for fe in fn.fes('sound')]) # doctest: +NORMALIZE_WHITESPACE\n        [('Cause_to_make_noise', 'Sound_maker'), ('Make_noise', 'Sound'),\n         ('Make_noise', 'Sound_source'), ('Sound_movement', 'Location_of_sound_source'),\n         ('Sound_movement', 'Sound'), ('Sound_movement', 'Sound_source'),\n         ('Sounds', 'Component_sound'), ('Sounds', 'Location_of_sound_source'),\n         ('Sounds', 'Sound_source'), ('Vocalizations', 'Location_of_sound_source'),\n         ('Vocalizations', 'Sound_source')]\n        >>> sorted([(fe.frame.name,fe.name) for fe in fn.fes('sound',r'(?i)make_noise')]) # doctest: +NORMALIZE_WHITESPACE\n        [('Cause_to_make_noise', 'Sound_maker'),\n         ('Make_noise', 'Sound'),\n         ('Make_noise', 'Sound_source')]\n        >>> sorted(set(fe.name for fe in fn.fes('^sound')))\n        ['Sound', 'Sound_maker', 'Sound_source']\n        >>> len(fn.fes('^sound$'))\n        2\n\n        :param name: A regular expression pattern used to match against\n            frame element names. If 'name' is None, then a list of all\n            frame elements will be returned.\n        :type name: str\n        :return: A list of matching frame elements\n        :rtype: list(AttrDict)\n        "
        if frame is not None:
            if isinstance(frame, int):
                frames = [self.frame(frame)]
            elif isinstance(frame, str):
                frames = self.frames(frame)
            else:
                frames = [frame]
        else:
            frames = self.frames()
        return PrettyList((fe for f in frames for (fename, fe) in f.FE.items() if name is None or re.search(name, fename, re.I)))

    def lus(self, name=None, frame=None):
        if False:
            while True:
                i = 10
        '\n        Obtain details for lexical units.\n        Optionally restrict by lexical unit name pattern, and/or to a certain frame\n        or frames whose name matches a pattern.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> len(fn.lus()) in (11829, 13572) # FN 1.5 and 1.7, resp.\n        True\n        >>> PrettyList(sorted(fn.lus(r\'(?i)a little\'), key=itemgetter(\'ID\')), maxReprSize=0, breakLines=True)\n        [<lu ID=14733 name=a little.n>,\n         <lu ID=14743 name=a little.adv>,\n         <lu ID=14744 name=a little bit.adv>]\n        >>> PrettyList(sorted(fn.lus(r\'interest\', r\'(?i)stimulus\'), key=itemgetter(\'ID\')))\n        [<lu ID=14894 name=interested.a>, <lu ID=14920 name=interesting.a>]\n\n        A brief intro to Lexical Units (excerpted from "FrameNet II:\n        Extended Theory and Practice" by Ruppenhofer et. al., 2010):\n\n        A lexical unit (LU) is a pairing of a word with a meaning. For\n        example, the "Apply_heat" Frame describes a common situation\n        involving a Cook, some Food, and a Heating Instrument, and is\n        _evoked_ by words such as bake, blanch, boil, broil, brown,\n        simmer, steam, etc. These frame-evoking words are the LUs in the\n        Apply_heat frame. Each sense of a polysemous word is a different\n        LU.\n\n        We have used the word "word" in talking about LUs. The reality\n        is actually rather complex. When we say that the word "bake" is\n        polysemous, we mean that the lemma "bake.v" (which has the\n        word-forms "bake", "bakes", "baked", and "baking") is linked to\n        three different frames:\n\n           - Apply_heat: "Michelle baked the potatoes for 45 minutes."\n\n           - Cooking_creation: "Michelle baked her mother a cake for her birthday."\n\n           - Absorb_heat: "The potatoes have to bake for more than 30 minutes."\n\n        These constitute three different LUs, with different\n        definitions.\n\n        Multiword expressions such as "given name" and hyphenated words\n        like "shut-eye" can also be LUs. Idiomatic phrases such as\n        "middle of nowhere" and "give the slip (to)" are also defined as\n        LUs in the appropriate frames ("Isolated_places" and "Evading",\n        respectively), and their internal structure is not analyzed.\n\n        Framenet provides multiple annotated examples of each sense of a\n        word (i.e. each LU).  Moreover, the set of examples\n        (approximately 20 per LU) illustrates all of the combinatorial\n        possibilities of the lexical unit.\n\n        Each LU is linked to a Frame, and hence to the other words which\n        evoke that Frame. This makes the FrameNet database similar to a\n        thesaurus, grouping together semantically similar words.\n\n        In the simplest case, frame-evoking words are verbs such as\n        "fried" in:\n\n           "Matilde fried the catfish in a heavy iron skillet."\n\n        Sometimes event nouns may evoke a Frame. For example,\n        "reduction" evokes "Cause_change_of_scalar_position" in:\n\n           "...the reduction of debt levels to $665 million from $2.6 billion."\n\n        Adjectives may also evoke a Frame. For example, "asleep" may\n        evoke the "Sleep" frame as in:\n\n           "They were asleep for hours."\n\n        Many common nouns, such as artifacts like "hat" or "tower",\n        typically serve as dependents rather than clearly evoking their\n        own frames.\n\n        :param name: A regular expression pattern used to search the LU\n            names. Note that LU names take the form of a dotted\n            string (e.g. "run.v" or "a little.adv") in which a\n            lemma precedes the "." and a POS follows the\n            dot. The lemma may be composed of a single lexeme\n            (e.g. "run") or of multiple lexemes (e.g. "a\n            little"). If \'name\' is not given, then all LUs will\n            be returned.\n\n            The valid POSes are:\n\n                   v    - verb\n                   n    - noun\n                   a    - adjective\n                   adv  - adverb\n                   prep - preposition\n                   num  - numbers\n                   intj - interjection\n                   art  - article\n                   c    - conjunction\n                   scon - subordinating conjunction\n\n        :type name: str\n        :type frame: str or int or frame\n        :return: A list of selected (or all) lexical units\n        :rtype: list of LU objects (dicts). See the lu() function for info\n          about the specifics of LU objects.\n\n        '
        if not self._lu_idx:
            self._buildluindex()
        if name is not None:
            result = PrettyList((self.lu(luID) for (luID, luName) in self.lu_ids_and_names(name).items()))
            if frame is not None:
                if isinstance(frame, int):
                    frameIDs = {frame}
                elif isinstance(frame, str):
                    frameIDs = {f.ID for f in self.frames(frame)}
                else:
                    frameIDs = {frame.ID}
                result = PrettyList((lu for lu in result if lu.frame.ID in frameIDs))
        elif frame is not None:
            if isinstance(frame, int):
                frames = [self.frame(frame)]
            elif isinstance(frame, str):
                frames = self.frames(frame)
            else:
                frames = [frame]
            result = PrettyLazyIteratorList(iter(LazyConcatenation((list(f.lexUnit.values()) for f in frames))))
        else:
            luIDs = [luID for (luID, lu) in self._lu_idx.items() if lu.status not in self._bad_statuses]
            result = PrettyLazyMap(self.lu, luIDs)
        return result

    def lu_ids_and_names(self, name=None):
        if False:
            print('Hello World!')
        '\n        Uses the LU index, which is much faster than looking up each LU definition\n        if only the names and IDs are needed.\n        '
        if not self._lu_idx:
            self._buildluindex()
        return {luID: luinfo.name for (luID, luinfo) in self._lu_idx.items() if luinfo.status not in self._bad_statuses and (name is None or re.search(name, luinfo.name) is not None)}

    def docs_metadata(self, name=None):
        if False:
            return 10
        '\n        Return an index of the annotated documents in Framenet.\n\n        Details for a specific annotated document can be obtained using this\n        class\'s doc() function and pass it the value of the \'ID\' field.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> len(fn.docs()) in (78, 107) # FN 1.5 and 1.7, resp.\n        True\n        >>> set([x.corpname for x in fn.docs_metadata()])>=set([\'ANC\', \'KBEval\',                     \'LUCorpus-v0.3\', \'Miscellaneous\', \'NTI\', \'PropBank\'])\n        True\n\n        :param name: A regular expression pattern used to search the\n            file name of each annotated document. The document\'s\n            file name contains the name of the corpus that the\n            document is from, followed by two underscores "__"\n            followed by the document name. So, for example, the\n            file name "LUCorpus-v0.3__20000410_nyt-NEW.xml" is\n            from the corpus named "LUCorpus-v0.3" and the\n            document name is "20000410_nyt-NEW.xml".\n        :type name: str\n        :return: A list of selected (or all) annotated documents\n        :rtype: list of dicts, where each dict object contains the following\n                keys:\n\n                - \'name\'\n                - \'ID\'\n                - \'corpid\'\n                - \'corpname\'\n                - \'description\'\n                - \'filename\'\n        '
        try:
            ftlist = PrettyList(self._fulltext_idx.values())
        except AttributeError:
            self._buildcorpusindex()
            ftlist = PrettyList(self._fulltext_idx.values())
        if name is None:
            return ftlist
        else:
            return PrettyList((x for x in ftlist if re.search(name, x['filename']) is not None))

    def docs(self, name=None):
        if False:
            while True:
                i = 10
        '\n        Return a list of the annotated full-text documents in FrameNet,\n        optionally filtered by a regex to be matched against the document name.\n        '
        return PrettyLazyMap(lambda x: self.doc(x.ID), self.docs_metadata(name))

    def sents(self, exemplars=True, full_text=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Annotated sentences matching the specified criteria.\n        '
        if exemplars:
            if full_text:
                return self.exemplars() + self.ft_sents()
            else:
                return self.exemplars()
        elif full_text:
            return self.ft_sents()

    def annotations(self, luNamePattern=None, exemplars=True, full_text=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Frame annotation sets matching the specified criteria.\n        '
        if exemplars:
            epart = PrettyLazyIteratorList((sent.frameAnnotation for sent in self.exemplars(luNamePattern)))
        else:
            epart = []
        if full_text:
            if luNamePattern is not None:
                matchedLUIDs = set(self.lu_ids_and_names(luNamePattern).keys())
            ftpart = PrettyLazyIteratorList((aset for sent in self.ft_sents() for aset in sent.annotationSet[1:] if luNamePattern is None or aset.get('luID', 'CXN_ASET') in matchedLUIDs))
        else:
            ftpart = []
        if exemplars:
            if full_text:
                return epart + ftpart
            else:
                return epart
        elif full_text:
            return ftpart

    def exemplars(self, luNamePattern=None, frame=None, fe=None, fe2=None):
        if False:
            i = 10
            return i + 15
        "\n        Lexicographic exemplar sentences, optionally filtered by LU name and/or 1-2 FEs that\n        are realized overtly. 'frame' may be a name pattern, frame ID, or frame instance.\n        'fe' may be a name pattern or FE instance; if specified, 'fe2' may also\n        be specified to retrieve sentences with both overt FEs (in either order).\n        "
        if fe is None and fe2 is not None:
            raise FramenetError('exemplars(..., fe=None, fe2=<value>) is not allowed')
        elif fe is not None and fe2 is not None:
            if not isinstance(fe2, str):
                if isinstance(fe, str):
                    (fe, fe2) = (fe2, fe)
                elif fe.frame is not fe2.frame:
                    raise FramenetError('exemplars() call with inconsistent `fe` and `fe2` specification (frames must match)')
        if frame is None and fe is not None and (not isinstance(fe, str)):
            frame = fe.frame
        lusByFrame = defaultdict(list)
        if frame is not None or luNamePattern is not None:
            if frame is None or isinstance(frame, str):
                if luNamePattern is not None:
                    frames = set()
                    for lu in self.lus(luNamePattern, frame=frame):
                        frames.add(lu.frame.ID)
                        lusByFrame[lu.frame.name].append(lu)
                    frames = LazyMap(self.frame, list(frames))
                else:
                    frames = self.frames(frame)
            else:
                if isinstance(frame, int):
                    frames = [self.frame(frame)]
                else:
                    frames = [frame]
                if luNamePattern is not None:
                    lusByFrame = {frame.name: self.lus(luNamePattern, frame=frame)}
            if fe is not None:
                if isinstance(fe, str):
                    frames = PrettyLazyIteratorList((f for f in frames if fe in f.FE or any((re.search(fe, ffe, re.I) for ffe in f.FE.keys()))))
                else:
                    if fe.frame not in frames:
                        raise FramenetError('exemplars() call with inconsistent `frame` and `fe` specification')
                    frames = [fe.frame]
                if fe2 is not None:
                    if isinstance(fe2, str):
                        frames = PrettyLazyIteratorList((f for f in frames if fe2 in f.FE or any((re.search(fe2, ffe, re.I) for ffe in f.FE.keys()))))
        elif fe is not None:
            frames = {ffe.frame.ID for ffe in self.fes(fe)}
            if fe2 is not None:
                frames2 = {ffe.frame.ID for ffe in self.fes(fe2)}
                frames = frames & frames2
            frames = LazyMap(self.frame, list(frames))
        else:
            frames = self.frames()

        def _matching_exs():
            if False:
                print('Hello World!')
            for f in frames:
                fes = fes2 = None
                if fe is not None:
                    fes = {ffe for ffe in f.FE.keys() if re.search(fe, ffe, re.I)} if isinstance(fe, str) else {fe.name}
                    if fe2 is not None:
                        fes2 = {ffe for ffe in f.FE.keys() if re.search(fe2, ffe, re.I)} if isinstance(fe2, str) else {fe2.name}
                for lu in lusByFrame[f.name] if luNamePattern is not None else f.lexUnit.values():
                    for ex in lu.exemplars:
                        if (fes is None or self._exemplar_of_fes(ex, fes)) and (fes2 is None or self._exemplar_of_fes(ex, fes2)):
                            yield ex
        return PrettyLazyIteratorList(_matching_exs())

    def _exemplar_of_fes(self, ex, fes=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Given an exemplar sentence and a set of FE names, return the subset of FE names\n        that are realized overtly in the sentence on the FE, FE2, or FE3 layer.\n\n        If 'fes' is None, returns all overt FE names.\n        "
        overtNames = set(list(zip(*ex.FE[0]))[2]) if ex.FE[0] else set()
        if 'FE2' in ex:
            overtNames |= set(list(zip(*ex.FE2[0]))[2]) if ex.FE2[0] else set()
            if 'FE3' in ex:
                overtNames |= set(list(zip(*ex.FE3[0]))[2]) if ex.FE3[0] else set()
        return overtNames & fes if fes is not None else overtNames

    def ft_sents(self, docNamePattern=None):
        if False:
            return 10
        '\n        Full-text annotation sentences, optionally filtered by document name.\n        '
        return PrettyLazyIteratorList((sent for d in self.docs(docNamePattern) for sent in d.sentence))

    def frame_relation_types(self):
        if False:
            i = 10
            return i + 15
        "\n        Obtain a list of frame relation types.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> frts = sorted(fn.frame_relation_types(), key=itemgetter('ID'))\n        >>> isinstance(frts, list)\n        True\n        >>> len(frts) in (9, 10)    # FN 1.5 and 1.7, resp.\n        True\n        >>> PrettyDict(frts[0], breakLines=True)\n        {'ID': 1,\n         '_type': 'framerelationtype',\n         'frameRelations': [<Parent=Event -- Inheritance -> Child=Change_of_consistency>, <Parent=Event -- Inheritance -> Child=Rotting>, ...],\n         'name': 'Inheritance',\n         'subFrameName': 'Child',\n         'superFrameName': 'Parent'}\n\n        :return: A list of all of the frame relation types in framenet\n        :rtype: list(dict)\n        "
        if not self._freltyp_idx:
            self._buildrelationindex()
        return self._freltyp_idx.values()

    def frame_relations(self, frame=None, frame2=None, type=None):
        if False:
            while True:
                i = 10
        "\n        :param frame: (optional) frame object, name, or ID; only relations involving\n            this frame will be returned\n        :param frame2: (optional; 'frame' must be a different frame) only show relations\n            between the two specified frames, in either direction\n        :param type: (optional) frame relation type (name or object); show only relations\n            of this type\n        :type frame: int or str or AttrDict\n        :return: A list of all of the frame relations in framenet\n        :rtype: list(dict)\n\n        >>> from nltk.corpus import framenet as fn\n        >>> frels = fn.frame_relations()\n        >>> isinstance(frels, list)\n        True\n        >>> len(frels) in (1676, 2070)  # FN 1.5 and 1.7, resp.\n        True\n        >>> PrettyList(fn.frame_relations('Cooking_creation'), maxReprSize=0, breakLines=True)\n        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>,\n         <Parent=Apply_heat -- Using -> Child=Cooking_creation>,\n         <MainEntry=Apply_heat -- See_also -> ReferringEntry=Cooking_creation>]\n        >>> PrettyList(fn.frame_relations(274), breakLines=True)\n        [<Parent=Avoiding -- Inheritance -> Child=Dodging>,\n         <Parent=Avoiding -- Inheritance -> Child=Evading>, ...]\n        >>> PrettyList(fn.frame_relations(fn.frame('Cooking_creation')), breakLines=True)\n        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>,\n         <Parent=Apply_heat -- Using -> Child=Cooking_creation>, ...]\n        >>> PrettyList(fn.frame_relations('Cooking_creation', type='Inheritance'))\n        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>]\n        >>> PrettyList(fn.frame_relations('Cooking_creation', 'Apply_heat'), breakLines=True) # doctest: +NORMALIZE_WHITESPACE\n        [<Parent=Apply_heat -- Using -> Child=Cooking_creation>,\n        <MainEntry=Apply_heat -- See_also -> ReferringEntry=Cooking_creation>]\n        "
        relation_type = type
        if not self._frel_idx:
            self._buildrelationindex()
        rels = None
        if relation_type is not None:
            if not isinstance(relation_type, dict):
                type = [rt for rt in self.frame_relation_types() if rt.name == type][0]
                assert isinstance(type, dict)
        if frame is not None:
            if isinstance(frame, dict) and 'frameRelations' in frame:
                rels = PrettyList(frame.frameRelations)
            else:
                if not isinstance(frame, int):
                    if isinstance(frame, dict):
                        frame = frame.ID
                    else:
                        frame = self.frame_by_name(frame).ID
                rels = [self._frel_idx[frelID] for frelID in self._frel_f_idx[frame]]
            if type is not None:
                rels = [rel for rel in rels if rel.type is type]
        elif type is not None:
            rels = type.frameRelations
        else:
            rels = self._frel_idx.values()
        if frame2 is not None:
            if frame is None:
                raise FramenetError('frame_relations(frame=None, frame2=<value>) is not allowed')
            if not isinstance(frame2, int):
                if isinstance(frame2, dict):
                    frame2 = frame2.ID
                else:
                    frame2 = self.frame_by_name(frame2).ID
            if frame == frame2:
                raise FramenetError('The two frame arguments to frame_relations() must be different frames')
            rels = [rel for rel in rels if rel.superFrame.ID == frame2 or rel.subFrame.ID == frame2]
        return PrettyList(sorted(rels, key=lambda frel: (frel.type.ID, frel.superFrameName, frel.subFrameName)))

    def fe_relations(self):
        if False:
            while True:
                i = 10
        "\n        Obtain a list of frame element relations.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> ferels = fn.fe_relations()\n        >>> isinstance(ferels, list)\n        True\n        >>> len(ferels) in (10020, 12393)   # FN 1.5 and 1.7, resp.\n        True\n        >>> PrettyDict(ferels[0], breakLines=True) # doctest: +NORMALIZE_WHITESPACE\n        {'ID': 14642,\n        '_type': 'ferelation',\n        'frameRelation': <Parent=Abounding_with -- Inheritance -> Child=Lively_place>,\n        'subFE': <fe ID=11370 name=Degree>,\n        'subFEName': 'Degree',\n        'subFrame': <frame ID=1904 name=Lively_place>,\n        'subID': 11370,\n        'supID': 2271,\n        'superFE': <fe ID=2271 name=Degree>,\n        'superFEName': 'Degree',\n        'superFrame': <frame ID=262 name=Abounding_with>,\n        'type': <framerelationtype ID=1 name=Inheritance>}\n\n        :return: A list of all of the frame element relations in framenet\n        :rtype: list(dict)\n        "
        if not self._ferel_idx:
            self._buildrelationindex()
        return PrettyList(sorted(self._ferel_idx.values(), key=lambda ferel: (ferel.type.ID, ferel.frameRelation.superFrameName, ferel.superFEName, ferel.frameRelation.subFrameName, ferel.subFEName)))

    def semtypes(self):
        if False:
            i = 10
            return i + 15
        "\n        Obtain a list of semantic types.\n\n        >>> from nltk.corpus import framenet as fn\n        >>> stypes = fn.semtypes()\n        >>> len(stypes) in (73, 109) # FN 1.5 and 1.7, resp.\n        True\n        >>> sorted(stypes[0].keys())\n        ['ID', '_type', 'abbrev', 'definition', 'definitionMarkup', 'name', 'rootType', 'subTypes', 'superType']\n\n        :return: A list of all of the semantic types in framenet\n        :rtype: list(dict)\n        "
        if not self._semtypes:
            self._loadsemtypes()
        return PrettyList((self._semtypes[i] for i in self._semtypes if isinstance(i, int)))

    def _load_xml_attributes(self, d, elt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Extracts a subset of the attributes from the given element and\n        returns them in a dictionary.\n\n        :param d: A dictionary in which to store the attributes.\n        :type d: dict\n        :param elt: An ElementTree Element\n        :type elt: Element\n        :return: Returns the input dict ``d`` possibly including attributes from ``elt``\n        :rtype: dict\n        '
        d = type(d)(d)
        try:
            attr_dict = elt.attrib
        except AttributeError:
            return d
        if attr_dict is None:
            return d
        ignore_attrs = ['xsi', 'schemaLocation', 'xmlns', 'bgColor', 'fgColor']
        for attr in attr_dict:
            if any((attr.endswith(x) for x in ignore_attrs)):
                continue
            val = attr_dict[attr]
            if val.isdigit():
                d[attr] = int(val)
            else:
                d[attr] = val
        return d

    def _strip_tags(self, data):
        if False:
            print('Hello World!')
        '\n        Gets rid of all tags and newline characters from the given input\n\n        :return: A cleaned-up version of the input string\n        :rtype: str\n        '
        try:
            "\n            # Look for boundary issues in markup. (Sometimes FEs are pluralized in definitions.)\n            m = re.search(r'\\w[<][^/]|[<][/][^>]+[>](s\\w|[a-rt-z0-9])', data)\n            if m:\n                print('Markup boundary:', data[max(0,m.start(0)-10):m.end(0)+10].replace('\\n',' '), file=sys.stderr)\n            "
            data = data.replace('<t>', '')
            data = data.replace('</t>', '')
            data = re.sub('<fex name="[^"]+">', '', data)
            data = data.replace('</fex>', '')
            data = data.replace('<fen>', '')
            data = data.replace('</fen>', '')
            data = data.replace('<m>', '')
            data = data.replace('</m>', '')
            data = data.replace('<ment>', '')
            data = data.replace('</ment>', '')
            data = data.replace('<ex>', "'")
            data = data.replace('</ex>', "'")
            data = data.replace('<gov>', '')
            data = data.replace('</gov>', '')
            data = data.replace('<x>', '')
            data = data.replace('</x>', '')
            data = data.replace('<def-root>', '')
            data = data.replace('</def-root>', '')
            data = data.replace('\n', ' ')
        except AttributeError:
            pass
        return data

    def _handle_elt(self, elt, tagspec=None):
        if False:
            while True:
                i = 10
        'Extracts and returns the attributes of the given element'
        return self._load_xml_attributes(AttrDict(), elt)

    def _handle_fulltextindex_elt(self, elt, tagspec=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Extracts corpus/document info from the fulltextIndex.xml file.\n\n        Note that this function "flattens" the information contained\n        in each of the "corpus" elements, so that each "document"\n        element will contain attributes for the corpus and\n        corpusid. Also, each of the "document" items will contain a\n        new attribute called "filename" that is the base file name of\n        the xml file for the document in the "fulltext" subdir of the\n        Framenet corpus.\n        '
        ftinfo = self._load_xml_attributes(AttrDict(), elt)
        corpname = ftinfo.name
        corpid = ftinfo.ID
        retlist = []
        for sub in elt:
            if sub.tag.endswith('document'):
                doc = self._load_xml_attributes(AttrDict(), sub)
                if 'name' in doc:
                    docname = doc.name
                else:
                    docname = doc.description
                doc.filename = f'{corpname}__{docname}.xml'
                doc.URL = self._fnweb_url + '/' + self._fulltext_dir + '/' + doc.filename
                doc.corpname = corpname
                doc.corpid = corpid
                retlist.append(doc)
        return retlist

    def _handle_frame_elt(self, elt, ignorekeys=[]):
        if False:
            return 10
        'Load the info for a Frame from a frame xml file'
        frinfo = self._load_xml_attributes(AttrDict(), elt)
        frinfo['_type'] = 'frame'
        frinfo['definition'] = ''
        frinfo['definitionMarkup'] = ''
        frinfo['FE'] = PrettyDict()
        frinfo['FEcoreSets'] = []
        frinfo['lexUnit'] = PrettyDict()
        frinfo['semTypes'] = []
        for k in ignorekeys:
            if k in frinfo:
                del frinfo[k]
        for sub in elt:
            if sub.tag.endswith('definition') and 'definition' not in ignorekeys:
                frinfo['definitionMarkup'] = sub.text
                frinfo['definition'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('FE') and 'FE' not in ignorekeys:
                feinfo = self._handle_fe_elt(sub)
                frinfo['FE'][feinfo.name] = feinfo
                feinfo['frame'] = frinfo
            elif sub.tag.endswith('FEcoreSet') and 'FEcoreSet' not in ignorekeys:
                coreset = self._handle_fecoreset_elt(sub)
                frinfo['FEcoreSets'].append(PrettyList((frinfo['FE'][fe.name] for fe in coreset)))
            elif sub.tag.endswith('lexUnit') and 'lexUnit' not in ignorekeys:
                luentry = self._handle_framelexunit_elt(sub)
                if luentry['status'] in self._bad_statuses:
                    continue
                luentry['frame'] = frinfo
                luentry['URL'] = self._fnweb_url + '/' + self._lu_dir + '/' + 'lu{}.xml'.format(luentry['ID'])
                luentry['subCorpus'] = Future((lambda lu: lambda : self._lu_file(lu).subCorpus)(luentry))
                luentry['exemplars'] = Future((lambda lu: lambda : self._lu_file(lu).exemplars)(luentry))
                frinfo['lexUnit'][luentry.name] = luentry
                if not self._lu_idx:
                    self._buildluindex()
                self._lu_idx[luentry.ID] = luentry
            elif sub.tag.endswith('semType') and 'semTypes' not in ignorekeys:
                semtypeinfo = self._load_xml_attributes(AttrDict(), sub)
                frinfo['semTypes'].append(self.semtype(semtypeinfo.ID))
        frinfo['frameRelations'] = self.frame_relations(frame=frinfo)
        for fe in frinfo.FE.values():
            if fe.requiresFE:
                (name, ID) = (fe.requiresFE.name, fe.requiresFE.ID)
                fe.requiresFE = frinfo.FE[name]
                assert fe.requiresFE.ID == ID
            if fe.excludesFE:
                (name, ID) = (fe.excludesFE.name, fe.excludesFE.ID)
                fe.excludesFE = frinfo.FE[name]
                assert fe.excludesFE.ID == ID
        return frinfo

    def _handle_fecoreset_elt(self, elt):
        if False:
            print('Hello World!')
        'Load fe coreset info from xml.'
        info = self._load_xml_attributes(AttrDict(), elt)
        tmp = []
        for sub in elt:
            tmp.append(self._load_xml_attributes(AttrDict(), sub))
        return tmp

    def _handle_framerelationtype_elt(self, elt, *args):
        if False:
            for i in range(10):
                print('nop')
        'Load frame-relation element and its child fe-relation elements from frRelation.xml.'
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'framerelationtype'
        info['frameRelations'] = PrettyList()
        for sub in elt:
            if sub.tag.endswith('frameRelation'):
                frel = self._handle_framerelation_elt(sub)
                frel['type'] = info
                for ferel in frel.feRelations:
                    ferel['type'] = info
                info['frameRelations'].append(frel)
        return info

    def _handle_framerelation_elt(self, elt):
        if False:
            return 10
        'Load frame-relation element and its child fe-relation elements from frRelation.xml.'
        info = self._load_xml_attributes(AttrDict(), elt)
        assert info['superFrameName'] != info['subFrameName'], (elt, info)
        info['_type'] = 'framerelation'
        info['feRelations'] = PrettyList()
        for sub in elt:
            if sub.tag.endswith('FERelation'):
                ferel = self._handle_elt(sub)
                ferel['_type'] = 'ferelation'
                ferel['frameRelation'] = info
                info['feRelations'].append(ferel)
        return info

    def _handle_fulltextannotation_elt(self, elt):
        if False:
            for i in range(10):
                print('nop')
        "Load full annotation info for a document from its xml\n        file. The main element (fullTextAnnotation) contains a 'header'\n        element (which we ignore here) and a bunch of 'sentence'\n        elements."
        info = AttrDict()
        info['_type'] = 'fulltext_annotation'
        info['sentence'] = []
        for sub in elt:
            if sub.tag.endswith('header'):
                continue
            elif sub.tag.endswith('sentence'):
                s = self._handle_fulltext_sentence_elt(sub)
                s.doc = info
                info['sentence'].append(s)
        return info

    def _handle_fulltext_sentence_elt(self, elt):
        if False:
            i = 10
            return i + 15
        'Load information from the given \'sentence\' element. Each\n        \'sentence\' element contains a "text" and "annotationSet" sub\n        elements.'
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'fulltext_sentence'
        info['annotationSet'] = []
        info['targets'] = []
        target_spans = set()
        info['_ascii'] = types.MethodType(_annotation_ascii, info)
        info['text'] = ''
        for sub in elt:
            if sub.tag.endswith('text'):
                info['text'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('annotationSet'):
                a = self._handle_fulltextannotationset_elt(sub, is_pos=len(info['annotationSet']) == 0)
                if 'cxnID' in a:
                    continue
                a.sent = info
                a.text = info.text
                info['annotationSet'].append(a)
                if 'Target' in a:
                    for tspan in a.Target:
                        if tspan in target_spans:
                            self._warn('Duplicate target span "{}"'.format(info.text[slice(*tspan)]), tspan, 'in sentence', info['ID'], info.text)
                        else:
                            target_spans.add(tspan)
                    info['targets'].append((a.Target, a.luName, a.frameName))
        assert info['annotationSet'][0].status == 'UNANN'
        info['POS'] = info['annotationSet'][0].POS
        info['POS_tagset'] = info['annotationSet'][0].POS_tagset
        return info

    def _handle_fulltextannotationset_elt(self, elt, is_pos=False):
        if False:
            for i in range(10):
                print('nop')
        'Load information from the given \'annotationSet\' element. Each\n        \'annotationSet\' contains several "layer" elements.'
        info = self._handle_luannotationset_elt(elt, is_pos=is_pos)
        if not is_pos:
            info['_type'] = 'fulltext_annotationset'
            if 'cxnID' not in info:
                info['LU'] = self.lu(info.luID, luName=info.luName, frameID=info.frameID, frameName=info.frameName)
                info['frame'] = info.LU.frame
        return info

    def _handle_fulltextlayer_elt(self, elt):
        if False:
            return 10
        'Load information from the given \'layer\' element. Each\n        \'layer\' contains several "label" elements.'
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'layer'
        info['label'] = []
        for sub in elt:
            if sub.tag.endswith('label'):
                l = self._load_xml_attributes(AttrDict(), sub)
                info['label'].append(l)
        return info

    def _handle_framelexunit_elt(self, elt):
        if False:
            while True:
                i = 10
        "Load the lexical unit info from an xml element in a frame's xml file."
        luinfo = AttrDict()
        luinfo['_type'] = 'lu'
        luinfo = self._load_xml_attributes(luinfo, elt)
        luinfo['definition'] = ''
        luinfo['definitionMarkup'] = ''
        luinfo['sentenceCount'] = PrettyDict()
        luinfo['lexemes'] = PrettyList()
        luinfo['semTypes'] = PrettyList()
        for sub in elt:
            if sub.tag.endswith('definition'):
                luinfo['definitionMarkup'] = sub.text
                luinfo['definition'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('sentenceCount'):
                luinfo['sentenceCount'] = self._load_xml_attributes(PrettyDict(), sub)
            elif sub.tag.endswith('lexeme'):
                lexemeinfo = self._load_xml_attributes(PrettyDict(), sub)
                if not isinstance(lexemeinfo.name, str):
                    lexemeinfo.name = str(lexemeinfo.name)
                luinfo['lexemes'].append(lexemeinfo)
            elif sub.tag.endswith('semType'):
                semtypeinfo = self._load_xml_attributes(PrettyDict(), sub)
                luinfo['semTypes'].append(self.semtype(semtypeinfo.ID))
        luinfo['lexemes'].sort(key=lambda x: x.order)
        return luinfo

    def _handle_lexunit_elt(self, elt, ignorekeys):
        if False:
            print('Hello World!')
        '\n        Load full info for a lexical unit from its xml file.\n        This should only be called when accessing corpus annotations\n        (which are not included in frame files).\n        '
        luinfo = self._load_xml_attributes(AttrDict(), elt)
        luinfo['_type'] = 'lu'
        luinfo['definition'] = ''
        luinfo['definitionMarkup'] = ''
        luinfo['subCorpus'] = PrettyList()
        luinfo['lexemes'] = PrettyList()
        luinfo['semTypes'] = PrettyList()
        for k in ignorekeys:
            if k in luinfo:
                del luinfo[k]
        for sub in elt:
            if sub.tag.endswith('header'):
                continue
            elif sub.tag.endswith('valences'):
                continue
            elif sub.tag.endswith('definition') and 'definition' not in ignorekeys:
                luinfo['definitionMarkup'] = sub.text
                luinfo['definition'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('subCorpus') and 'subCorpus' not in ignorekeys:
                sc = self._handle_lusubcorpus_elt(sub)
                if sc is not None:
                    luinfo['subCorpus'].append(sc)
            elif sub.tag.endswith('lexeme') and 'lexeme' not in ignorekeys:
                luinfo['lexemes'].append(self._load_xml_attributes(PrettyDict(), sub))
            elif sub.tag.endswith('semType') and 'semType' not in ignorekeys:
                semtypeinfo = self._load_xml_attributes(AttrDict(), sub)
                luinfo['semTypes'].append(self.semtype(semtypeinfo.ID))
        return luinfo

    def _handle_lusubcorpus_elt(self, elt):
        if False:
            return 10
        'Load a subcorpus of a lexical unit from the given xml.'
        sc = AttrDict()
        try:
            sc['name'] = elt.get('name')
        except AttributeError:
            return None
        sc['_type'] = 'lusubcorpus'
        sc['sentence'] = []
        for sub in elt:
            if sub.tag.endswith('sentence'):
                s = self._handle_lusentence_elt(sub)
                if s is not None:
                    sc['sentence'].append(s)
        return sc

    def _handle_lusentence_elt(self, elt):
        if False:
            print('Hello World!')
        'Load a sentence from a subcorpus of an LU from xml.'
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'lusentence'
        info['annotationSet'] = []
        info['_ascii'] = types.MethodType(_annotation_ascii, info)
        for sub in elt:
            if sub.tag.endswith('text'):
                info['text'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('annotationSet'):
                annset = self._handle_luannotationset_elt(sub, is_pos=len(info['annotationSet']) == 0)
                if annset is not None:
                    assert annset.status == 'UNANN' or 'FE' in annset, annset
                    if annset.status != 'UNANN':
                        info['frameAnnotation'] = annset
                    for k in ('Target', 'FE', 'FE2', 'FE3', 'GF', 'PT', 'POS', 'POS_tagset', 'Other', 'Sent', 'Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art'):
                        if k in annset:
                            info[k] = annset[k]
                    info['annotationSet'].append(annset)
                    annset['sent'] = info
                    annset['text'] = info.text
        return info

    def _handle_luannotationset_elt(self, elt, is_pos=False):
        if False:
            return 10
        'Load an annotation set from a sentence in an subcorpus of an LU'
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'posannotationset' if is_pos else 'luannotationset'
        info['layer'] = []
        info['_ascii'] = types.MethodType(_annotation_ascii, info)
        if 'cxnID' in info:
            return info
        for sub in elt:
            if sub.tag.endswith('layer'):
                l = self._handle_lulayer_elt(sub)
                if l is not None:
                    overt = []
                    ni = {}
                    info['layer'].append(l)
                    for lbl in l.label:
                        if 'start' in lbl:
                            thespan = (lbl.start, lbl.end + 1, lbl.name)
                            if l.name not in ('Sent', 'Other'):
                                assert thespan not in overt, (info.ID, l.name, thespan)
                            overt.append(thespan)
                        elif lbl.name in ni:
                            self._warn('FE with multiple NI entries:', lbl.name, ni[lbl.name], lbl.itype)
                        else:
                            ni[lbl.name] = lbl.itype
                    overt = sorted(overt)
                    if l.name == 'Target':
                        if not overt:
                            self._warn('Skipping empty Target layer in annotation set ID={}'.format(info.ID))
                            continue
                        assert all((lblname == 'Target' for (i, j, lblname) in overt))
                        if 'Target' in info:
                            self._warn('Annotation set {} has multiple Target layers'.format(info.ID))
                        else:
                            info['Target'] = [(i, j) for (i, j, _) in overt]
                    elif l.name == 'FE':
                        if l.rank == 1:
                            assert 'FE' not in info
                            info['FE'] = (overt, ni)
                        else:
                            assert 2 <= l.rank <= 3, l.rank
                            k = 'FE' + str(l.rank)
                            assert k not in info
                            info[k] = (overt, ni)
                    elif l.name in ('GF', 'PT'):
                        assert l.rank == 1
                        info[l.name] = overt
                    elif l.name in ('BNC', 'PENN'):
                        assert l.rank == 1
                        info['POS'] = overt
                        info['POS_tagset'] = l.name
                    else:
                        if is_pos:
                            if l.name not in ('NER', 'WSL'):
                                self._warn('Unexpected layer in sentence annotationset:', l.name)
                        elif l.name not in ('Sent', 'Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art', 'Other'):
                            self._warn('Unexpected layer in frame annotationset:', l.name)
                        info[l.name] = overt
        if not is_pos and 'cxnID' not in info:
            if 'Target' not in info:
                self._warn(f'Missing target in annotation set ID={info.ID}')
            assert 'FE' in info
            if 'FE3' in info:
                assert 'FE2' in info
        return info

    def _handle_lulayer_elt(self, elt):
        if False:
            return 10
        'Load a layer from an annotation set'
        layer = self._load_xml_attributes(AttrDict(), elt)
        layer['_type'] = 'lulayer'
        layer['label'] = []
        for sub in elt:
            if sub.tag.endswith('label'):
                l = self._load_xml_attributes(AttrDict(), sub)
                if l is not None:
                    layer['label'].append(l)
        return layer

    def _handle_fe_elt(self, elt):
        if False:
            return 10
        feinfo = self._load_xml_attributes(AttrDict(), elt)
        feinfo['_type'] = 'fe'
        feinfo['definition'] = ''
        feinfo['definitionMarkup'] = ''
        feinfo['semType'] = None
        feinfo['requiresFE'] = None
        feinfo['excludesFE'] = None
        for sub in elt:
            if sub.tag.endswith('definition'):
                feinfo['definitionMarkup'] = sub.text
                feinfo['definition'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('semType'):
                stinfo = self._load_xml_attributes(AttrDict(), sub)
                feinfo['semType'] = self.semtype(stinfo.ID)
            elif sub.tag.endswith('requiresFE'):
                feinfo['requiresFE'] = self._load_xml_attributes(AttrDict(), sub)
            elif sub.tag.endswith('excludesFE'):
                feinfo['excludesFE'] = self._load_xml_attributes(AttrDict(), sub)
        return feinfo

    def _handle_semtype_elt(self, elt, tagspec=None):
        if False:
            for i in range(10):
                print('nop')
        semt = self._load_xml_attributes(AttrDict(), elt)
        semt['_type'] = 'semtype'
        semt['superType'] = None
        semt['subTypes'] = PrettyList()
        for sub in elt:
            if sub.text is not None:
                semt['definitionMarkup'] = sub.text
                semt['definition'] = self._strip_tags(sub.text)
            else:
                supertypeinfo = self._load_xml_attributes(AttrDict(), sub)
                semt['superType'] = supertypeinfo
        return semt

def demo():
    if False:
        i = 10
        return i + 15
    from nltk.corpus import framenet as fn
    print('Building the indexes...')
    fn.buildindexes()
    print('Number of Frames:', len(fn.frames()))
    print('Number of Lexical Units:', len(fn.lus()))
    print('Number of annotated documents:', len(fn.docs()))
    print()
    print('getting frames whose name matches the (case insensitive) regex: "(?i)medical"')
    medframes = fn.frames('(?i)medical')
    print(f'Found {len(medframes)} Frames whose name matches "(?i)medical":')
    print([(f.name, f.ID) for f in medframes])
    tmp_id = medframes[0].ID
    m_frame = fn.frame(tmp_id)
    print('\nNumber of frame relations for the "{}" ({}) frame:'.format(m_frame.name, m_frame.ID), len(m_frame.frameRelations))
    for fr in m_frame.frameRelations:
        print('   ', fr)
    print(f'\nNumber of Frame Elements in the "{m_frame.name}" frame:', len(m_frame.FE))
    print('   ', [x for x in m_frame.FE])
    print(f'\nThe "core" Frame Elements in the "{m_frame.name}" frame:')
    print('   ', [x.name for x in m_frame.FE.values() if x.coreType == 'Core'])
    print('\nAll Lexical Units that are incorporated in the "Ailment" FE:')
    m_frame = fn.frame(239)
    ailment_lus = [x for x in m_frame.lexUnit.values() if 'incorporatedFE' in x and x.incorporatedFE == 'Ailment']
    print('   ', [x.name for x in ailment_lus])
    print(f'\nNumber of Lexical Units in the "{m_frame.name}" frame:', len(m_frame.lexUnit))
    print('  ', [x.name for x in m_frame.lexUnit.values()][:5], '...')
    tmp_id = m_frame.lexUnit['ailment.n'].ID
    luinfo = fn.lu_basic(tmp_id)
    print(f'\nInformation on the LU: {luinfo.name}')
    pprint(luinfo)
    print('\nNames of all of the corpora used for fulltext annotation:')
    allcorpora = {x.corpname for x in fn.docs_metadata()}
    pprint(list(allcorpora))
    firstcorp = list(allcorpora)[0]
    firstcorp_docs = fn.docs(firstcorp)
    print(f'\nNames of the annotated documents in the "{firstcorp}" corpus:')
    pprint([x.filename for x in firstcorp_docs])
    print('\nSearching for all Frames that have a lemma that matches the regexp: "^run.v$":')
    pprint(fn.frames_by_lemma('^run.v$'))
if __name__ == '__main__':
    demo()