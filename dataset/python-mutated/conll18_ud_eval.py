from __future__ import division
from __future__ import print_function
import argparse
import io
import sys
import unicodedata
import unittest
(ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC) = range(10)
CONTENT_DEPRELS = {'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl', 'vocative', 'expl', 'dislocated', 'advcl', 'advmod', 'discourse', 'nmod', 'appos', 'nummod', 'acl', 'amod', 'conj', 'fixed', 'flat', 'compound', 'list', 'parataxis', 'orphan', 'goeswith', 'reparandum', 'root', 'dep'}
FUNCTIONAL_DEPRELS = {'aux', 'cop', 'mark', 'det', 'clf', 'case', 'cc'}
UNIVERSAL_FEATURES = {'PronType', 'NumType', 'Poss', 'Reflex', 'Foreign', 'Abbr', 'Gender', 'Animacy', 'Number', 'Case', 'Definite', 'Degree', 'VerbForm', 'Mood', 'Tense', 'Aspect', 'Voice', 'Evident', 'Polarity', 'Person', 'Polite'}

class UDError(Exception):
    pass

def _decode(text):
    if False:
        while True:
            i = 10
    return text if sys.version_info[0] >= 3 or not isinstance(text, str) else text.decode('utf-8')

def _encode(text):
    if False:
        i = 10
        return i + 15
    return text if sys.version_info[0] >= 3 or not isinstance(text, unicode) else text.encode('utf-8')
CASE_DEPRELS = {'obl', 'nmod', 'conj', 'advcl'}
UNIVERSAL_DEPREL_EXTENSIONS = {'pass', 'relcl', 'xsubj'}

def process_enhanced_deps(deps):
    if False:
        while True:
            i = 10
    edeps = []
    if deps != '' and deps != '_':
        for edep in deps.split('|'):
            (hd, path) = edep.split(':', 1)
            steps = path.split('>')
            edeps.append((hd, steps))
    return edeps

def load_conllu(file, treebank_type):
    if False:
        return 10

    class UDRepresentation:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.characters = []
            self.tokens = []
            self.words = []
            self.sentences = []

    class UDSpan:

        def __init__(self, start, end):
            if False:
                while True:
                    i = 10
            self.start = start
            self.end = end

    class UDWord:

        def __init__(self, span, columns, is_multiword):
            if False:
                for i in range(10):
                    print('nop')
            self.span = span
            self.columns = columns
            self.is_multiword = is_multiword
            self.parent = None
            self.functional_children = []
            self.columns[FEATS] = '|'.join(sorted((feat for feat in columns[FEATS].split('|') if feat.split('=', 1)[0] in UNIVERSAL_FEATURES)))
            self.columns[DEPREL] = columns[DEPREL].split(':')[0]
            self.is_content_deprel = self.columns[DEPREL] in CONTENT_DEPRELS
            self.is_functional_deprel = self.columns[DEPREL] in FUNCTIONAL_DEPRELS
            self.columns[DEPS] = process_enhanced_deps(columns[DEPS])
    ud = UDRepresentation()
    (index, sentence_start) = (0, None)
    line_idx = 0
    while True:
        line = file.readline()
        line_idx += 1
        if not line:
            break
        line = _decode(line.rstrip('\r\n'))
        if sentence_start is None:
            if line.startswith('#'):
                continue
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)
        if not line:

            def process_word(word):
                if False:
                    for i in range(10):
                        print('nop')
                if word.parent == 'remapping':
                    raise UDError('There is a cycle in the sentence that ends at line %d' % line_idx)
                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head < 0 or head > len(ud.words) - sentence_start:
                        raise UDError("HEAD '{}' points outside of the sentence that ends at line {}".format(_encode(word.columns[HEAD]), line_idx))
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = 'remapping'
                        process_word(parent)
                        word.parent = parent
            position = sentence_start
            for word in ud.words[sentence_start:]:
                process_word(word)
                enhanced_deps = word.columns[DEPS]
                processed_deps = []
                for (head, steps) in word.columns[DEPS]:
                    if '.' in head:
                        if treebank_type.get('no_empty_nodes', False):
                            raise UDError('The collapsed CoNLL-U file still contains references to empty nodes at line {}: {}'.format(line_idx, _encode(line)))
                        else:
                            continue
                    hd = int(head)
                    parent = ud.words[sentence_start + hd - 1] if hd else hd
                    processed_deps.append((parent, steps))
                enhanced_deps = processed_deps
                if treebank_type.get('no_gapping', False):
                    processed_deps = []
                    for (parent, steps) in enhanced_deps:
                        if len(steps) > 1:
                            processed_deps.append((word.parent, [word.columns[DEPREL]]))
                        elif (parent, steps) in processed_deps:
                            True
                        else:
                            processed_deps.append((parent, steps))
                    enhanced_deps = processed_deps
                if treebank_type.get('no_shared_parents_in_coordination', False):
                    for (hd, steps) in enhanced_deps:
                        if len(steps) == 1 and steps[0].startswith('conj'):
                            enhanced_deps = [(hd, steps)]
                if treebank_type.get('no_shared_dependents_in_coordination', False):
                    processed_deps = []
                    for (hd, steps) in enhanced_deps:
                        duplicate = 0
                        for (hd2, steps2) in enhanced_deps:
                            if steps == steps2 and hd2 == word.columns[HEAD] and (hd != hd2):
                                duplicate = 1
                        if not duplicate:
                            processed_deps.append((hd, steps))
                    enhanced_deps = processed_deps
                if treebank_type.get('no_control', False):
                    processed_deps = []
                    for (parent, steps) in enhanced_deps:
                        include = 1
                        if parent and parent.columns[DEPREL] == 'xcomp':
                            for rel in steps:
                                if rel.startswith('nsubj'):
                                    include = 0
                        if include:
                            processed_deps.append((parent, steps))
                    enhanced_deps = processed_deps
                if treebank_type.get('no_external_arguments_of_relative_clauses', False):
                    processed_deps = []
                    for (parent, steps) in enhanced_deps:
                        if steps[0] == 'ref':
                            processed_deps.append((word.parent, [word.columns[DEPREL]]))
                        elif parent and parent.columns[DEPREL].startswith('acl') and (int(parent.columns[HEAD]) == position - sentence_start):
                            True
                        else:
                            processed_deps.append((parent, steps))
                    enhanced_deps = processed_deps
                if treebank_type.get('no_case_info', False):
                    processed_deps = []
                    for (hd, steps) in enhanced_deps:
                        processed_steps = []
                        for dep in steps:
                            depparts = dep.split(':')
                            if depparts[0] in CASE_DEPRELS:
                                if len(depparts) == 2 and (not depparts[1] in UNIVERSAL_DEPREL_EXTENSIONS):
                                    dep = depparts[0]
                            processed_steps.append(dep)
                        processed_deps.append((hd, processed_steps))
                    enhanced_deps = processed_deps
                position += 1
                word.columns[DEPS] = enhanced_deps
            for word in ud.words[sentence_start:]:
                if word.parent and word.is_functional_deprel:
                    word.parent.functional_children.append(word)
            if len(ud.words) == sentence_start:
                raise UDError('There is a sentence with 0 tokens (possibly a double blank line) at line %d' % line_idx)
            if len([word for word in ud.words[sentence_start:] if word.parent is None]) == 0:
                raise UDError('There are no roots in the sentence that ends at %d' % line_idx)
            if not treebank_type.get('multiple_roots_okay', False):
                if len([word for word in ud.words[sentence_start:] if word.parent is None]) > 1:
                    raise UDError('There are multiple roots in the sentence that ends at %d' % line_idx)
            ud.sentences[-1].end = index
            sentence_start = None
            continue
        columns = line.split('\t')
        if len(columns) != 10:
            raise UDError("The CoNLL-U line does not contain 10 tab-separated columns at line {}: '{}'".format(line_idx, _encode(line)))
        if '.' in columns[ID]:
            if treebank_type.get('no_empty_nodes', False):
                raise UDError('The collapsed CoNLL-U line still contains empty nodes at line {}: {}'.format(line_idx, _encode(line)))
            else:
                continue
        columns[FORM] = ''.join(filter(lambda c: unicodedata.category(c) != 'Zs', columns[FORM]))
        if not columns[FORM]:
            raise UDError('There is an empty FORM in the CoNLL-U file at line %d' % line_idx)
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])
        if '-' in columns[ID]:
            try:
                (start, end) = map(int, columns[ID].split('-'))
            except:
                raise UDError("Cannot parse multi-word token ID '{}' at line {}".format(_encode(columns[ID]), line_idx))
            for _ in range(start, end + 1):
                word_line = _decode(file.readline().rstrip('\r\n'))
                line_idx += 1
                word_columns = word_line.split('\t')
                if len(word_columns) != 10:
                    raise UDError("The CoNLL-U line does not contain 10 tab-separated columns at line {}: '{}'".format(line_idx, _encode(word_line)))
                ud.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        else:
            try:
                word_id = int(columns[ID])
            except:
                raise UDError("Cannot parse word ID '{}' at line {}".format(_encode(columns[ID]), line_idx))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected '{}' at line {}".format(_encode(columns[ID]), _encode(columns[FORM]), len(ud.words) - sentence_start + 1, line_idx))
            try:
                head_id = int(columns[HEAD])
            except ValueError as e:
                raise UDError("Cannot parse HEAD '{}' at line {}".format(_encode(columns[HEAD]), line_idx)) from e
            if head_id < 0:
                raise UDError('HEAD cannot be negative at line %d' % line_idx)
            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))
    if sentence_start is not None:
        raise UDError('The CoNLL-U file does not end with empty line')
    return ud

def evaluate(gold_ud, system_ud):
    if False:
        return 10

    class Score:

        def __init__(self, gold_total, system_total, correct, aligned_total=None):
            if False:
                print('Hello World!')
            self.correct = correct
            self.gold_total = gold_total
            self.system_total = system_total
            self.aligned_total = aligned_total
            self.precision = correct / system_total if system_total else 0.0
            self.recall = correct / gold_total if gold_total else 0.0
            self.f1 = 2 * correct / (system_total + gold_total) if system_total + gold_total else 0.0
            self.aligned_accuracy = correct / aligned_total if aligned_total else aligned_total

    class AlignmentWord:

        def __init__(self, gold_word, system_word):
            if False:
                return 10
            self.gold_word = gold_word
            self.system_word = system_word

    class Alignment:

        def __init__(self, gold_words, system_words):
            if False:
                while True:
                    i = 10
            self.gold_words = gold_words
            self.system_words = system_words
            self.matched_words = []
            self.matched_words_map = {}

        def append_aligned_words(self, gold_word, system_word):
            if False:
                print('Hello World!')
            self.matched_words.append(AlignmentWord(gold_word, system_word))
            self.matched_words_map[system_word] = gold_word

    def spans_score(gold_spans, system_spans):
        if False:
            print('Hello World!')
        (correct, gi, si) = (0, 0, 0)
        while gi < len(gold_spans) and si < len(system_spans):
            if system_spans[si].start < gold_spans[gi].start:
                si += 1
            elif gold_spans[gi].start < system_spans[si].start:
                gi += 1
            else:
                correct += gold_spans[gi].end == system_spans[si].end
                si += 1
                gi += 1
        return Score(len(gold_spans), len(system_spans), correct)

    def alignment_score(alignment, key_fn=None, filter_fn=None):
        if False:
            for i in range(10):
                print('nop')
        if filter_fn is not None:
            gold = sum((1 for gold in alignment.gold_words if filter_fn(gold)))
            system = sum((1 for system in alignment.system_words if filter_fn(system)))
            aligned = sum((1 for word in alignment.matched_words if filter_fn(word.gold_word)))
        else:
            gold = len(alignment.gold_words)
            system = len(alignment.system_words)
            aligned = len(alignment.matched_words)
        if key_fn is None:
            return Score(gold, system, aligned)

        def gold_aligned_gold(word):
            if False:
                while True:
                    i = 10
            return word

        def gold_aligned_system(word):
            if False:
                i = 10
                return i + 15
            return alignment.matched_words_map.get(word, 'NotAligned') if word is not None else None
        correct = 0
        for words in alignment.matched_words:
            if filter_fn is None or filter_fn(words.gold_word):
                if key_fn(words.gold_word, gold_aligned_gold) == key_fn(words.system_word, gold_aligned_system):
                    correct += 1
        return Score(gold, system, correct, aligned)

    def enhanced_alignment_score(alignment, EULAS):
        if False:
            while True:
                i = 10
        gold = 0
        for gold_word in alignment.gold_words:
            gold += len(gold_word.columns[DEPS])
        system = 0
        for system_word in alignment.system_words:
            system += len(system_word.columns[DEPS])
        correct = 0
        for words in alignment.matched_words:
            gold_deps = words.gold_word.columns[DEPS]
            system_deps = words.system_word.columns[DEPS]
            for (parent, dep) in gold_deps:
                eulas_dep = [d.split(':')[0] for d in dep]
                for (sparent, sdep) in system_deps:
                    eulas_sdep = [d.split(':')[0] for d in sdep]
                    if dep == sdep or (eulas_dep == eulas_sdep and EULAS):
                        if parent == alignment.matched_words_map.get(sparent, 'NotAligned'):
                            correct += 1
                        elif parent == 0 and sparent == 0:
                            correct += 1
        return Score(gold, system, correct)

    def beyond_end(words, i, multiword_span_end):
        if False:
            i = 10
            return i + 15
        if i >= len(words):
            return True
        if words[i].is_multiword:
            return words[i].span.start >= multiword_span_end
        return words[i].span.end > multiword_span_end

    def extend_end(word, multiword_span_end):
        if False:
            return 10
        if word.is_multiword and word.span.end > multiword_span_end:
            return word.span.end
        return multiword_span_end

    def find_multiword_span(gold_words, system_words, gi, si):
        if False:
            for i in range(10):
                print('nop')
        if gold_words[gi].is_multiword:
            multiword_span_end = gold_words[gi].span.end
            if not system_words[si].is_multiword and system_words[si].span.start < gold_words[gi].span.start:
                si += 1
        else:
            multiword_span_end = system_words[si].span.end
            if not gold_words[gi].is_multiword and gold_words[gi].span.start < system_words[si].span.start:
                gi += 1
        (gs, ss) = (gi, si)
        while not beyond_end(gold_words, gi, multiword_span_end) or not beyond_end(system_words, si, multiword_span_end):
            if gi < len(gold_words) and (si >= len(system_words) or gold_words[gi].span.start <= system_words[si].span.start):
                multiword_span_end = extend_end(gold_words[gi], multiword_span_end)
                gi += 1
            else:
                multiword_span_end = extend_end(system_words[si], multiword_span_end)
                si += 1
        return (gs, ss, gi, si)

    def compute_lcs(gold_words, system_words, gi, si, gs, ss):
        if False:
            for i in range(10):
                print('nop')
        lcs = [[0] * (si - ss) for i in range(gi - gs)]
        for g in reversed(range(gi - gs)):
            for s in reversed(range(si - ss)):
                if gold_words[gs + g].columns[FORM].lower() == system_words[ss + s].columns[FORM].lower():
                    lcs[g][s] = 1 + (lcs[g + 1][s + 1] if g + 1 < gi - gs and s + 1 < si - ss else 0)
                lcs[g][s] = max(lcs[g][s], lcs[g + 1][s] if g + 1 < gi - gs else 0)
                lcs[g][s] = max(lcs[g][s], lcs[g][s + 1] if s + 1 < si - ss else 0)
        return lcs

    def align_words(gold_words, system_words):
        if False:
            for i in range(10):
                print('nop')
        alignment = Alignment(gold_words, system_words)
        (gi, si) = (0, 0)
        while gi < len(gold_words) and si < len(system_words):
            if gold_words[gi].is_multiword or system_words[si].is_multiword:
                (gs, ss, gi, si) = find_multiword_span(gold_words, system_words, gi, si)
                if si > ss and gi > gs:
                    lcs = compute_lcs(gold_words, system_words, gi, si, gs, ss)
                    (s, g) = (0, 0)
                    while g < gi - gs and s < si - ss:
                        if gold_words[gs + g].columns[FORM].lower() == system_words[ss + s].columns[FORM].lower():
                            alignment.append_aligned_words(gold_words[gs + g], system_words[ss + s])
                            g += 1
                            s += 1
                        elif lcs[g][s] == (lcs[g + 1][s] if g + 1 < gi - gs else 0):
                            g += 1
                        else:
                            s += 1
            elif (gold_words[gi].span.start, gold_words[gi].span.end) == (system_words[si].span.start, system_words[si].span.end):
                alignment.append_aligned_words(gold_words[gi], system_words[si])
                gi += 1
                si += 1
            elif gold_words[gi].span.start <= system_words[si].span.start:
                gi += 1
            else:
                si += 1
        return alignment
    if gold_ud.characters != system_ud.characters:
        index = 0
        while index < len(gold_ud.characters) and index < len(system_ud.characters) and (gold_ud.characters[index] == system_ud.characters[index]):
            index += 1
        raise UDError('The concatenation of tokens in gold file and in system file differ!\n' + "First 20 differing characters in gold file: '{}' and system file: '{}'".format(''.join(map(_encode, gold_ud.characters[index:index + 20])), ''.join(map(_encode, system_ud.characters[index:index + 20]))))
    alignment = align_words(gold_ud.words, system_ud.words)
    return {'Tokens': spans_score(gold_ud.tokens, system_ud.tokens), 'Sentences': spans_score(gold_ud.sentences, system_ud.sentences), 'Words': alignment_score(alignment), 'UPOS': alignment_score(alignment, lambda w, _: w.columns[UPOS]), 'XPOS': alignment_score(alignment, lambda w, _: w.columns[XPOS]), 'UFeats': alignment_score(alignment, lambda w, _: w.columns[FEATS]), 'AllTags': alignment_score(alignment, lambda w, _: (w.columns[UPOS], w.columns[XPOS], w.columns[FEATS])), 'Lemmas': alignment_score(alignment, lambda w, ga: w.columns[LEMMA] if ga(w).columns[LEMMA] != '_' else '_'), 'UAS': alignment_score(alignment, lambda w, ga: ga(w.parent)), 'LAS': alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL])), 'ELAS': enhanced_alignment_score(alignment, 0), 'EULAS': enhanced_alignment_score(alignment, 1), 'CLAS': alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL]), filter_fn=lambda w: w.is_content_deprel), 'MLAS': alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL], w.columns[UPOS], w.columns[FEATS], [(ga(c), c.columns[DEPREL], c.columns[UPOS], c.columns[FEATS]) for c in w.functional_children]), filter_fn=lambda w: w.is_content_deprel), 'BLEX': alignment_score(alignment, lambda w, ga: (ga(w.parent), w.columns[DEPREL], w.columns[LEMMA] if ga(w).columns[LEMMA] != '_' else '_'), filter_fn=lambda w: w.is_content_deprel)}

def load_conllu_file(path, treebank_type=None):
    if False:
        while True:
            i = 10
    if treebank_type is None:
        treebank_type = {}
    _file = open(path, mode='r', **{'encoding': 'utf-8'} if sys.version_info >= (3, 0) else {})
    return load_conllu(_file, treebank_type)

def evaluate_wrapper(args):
    if False:
        i = 10
        return i + 15
    treebank_type = {}
    enhancements = list(args.enhancements)
    treebank_type['no_gapping'] = 1 if '1' in enhancements else 0
    treebank_type['no_shared_parents_in_coordination'] = 1 if '2' in enhancements else 0
    treebank_type['no_shared_dependents_in_coordination'] = 1 if '3' in enhancements else 0
    treebank_type['no_control'] = 1 if '4' in enhancements else 0
    treebank_type['no_external_arguments_of_relative_clauses'] = 1 if '5' in enhancements else 0
    treebank_type['no_case_info'] = 1 if '6' in enhancements else 0
    treebank_type['no_empty_nodes'] = args.no_empty_nodes
    treebank_type['multiple_roots_okay'] = args.multiple_roots_okay
    gold_ud = load_conllu_file(args.gold_file, treebank_type)
    system_ud = load_conllu_file(args.system_file, treebank_type)
    return evaluate(gold_ud, system_ud)

def build_evaluation_table(evaluation, verbose, counts, enhanced):
    if False:
        while True:
            i = 10
    text = []
    if not verbose and (not counts):
        text.append('LAS F1 Score: {:.2f}'.format(100 * evaluation['LAS'].f1))
        text.append('MLAS Score: {:.2f}'.format(100 * evaluation['MLAS'].f1))
        text.append('BLEX Score: {:.2f}'.format(100 * evaluation['BLEX'].f1))
        if enhanced:
            text.append('ELAS F1 Score: {:.2f}'.format(100 * evaluation['ELAS'].f1))
            text.append('EULAS F1 Score: {:.2f}'.format(100 * evaluation['EULAS'].f1))
    else:
        if counts:
            text.append('Metric     | Correct   |      Gold | Predicted | Aligned')
        else:
            text.append('Metric     | Precision |    Recall |  F1 Score | AligndAcc')
        text.append('-----------+-----------+-----------+-----------+-----------')
        metrics = ['Tokens', 'Sentences', 'Words', 'UPOS', 'XPOS', 'UFeats', 'AllTags', 'Lemmas', 'UAS', 'LAS', 'CLAS', 'MLAS', 'BLEX']
        if enhanced:
            metrics += ['ELAS', 'EULAS']
        for metric in metrics:
            if counts:
                text.append('{:11}|{:10} |{:10} |{:10} |{:10}'.format(metric, evaluation[metric].correct, evaluation[metric].gold_total, evaluation[metric].system_total, evaluation[metric].aligned_total or (evaluation[metric].correct if metric == 'Words' else '')))
            else:
                text.append('{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}'.format(metric, 100 * evaluation[metric].precision, 100 * evaluation[metric].recall, 100 * evaluation[metric].f1, '{:10.2f}'.format(100 * evaluation[metric].aligned_accuracy) if evaluation[metric].aligned_accuracy is not None else ''))
    return '\n'.join(text)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('gold_file', type=str, help='Name of the CoNLL-U file with the gold data.')
    parser.add_argument('system_file', type=str, help='Name of the CoNLL-U file with the predicted data.')
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help='Print all metrics.')
    parser.add_argument('--counts', '-c', default=False, action='store_true', help='Print raw counts of correct/gold/system/aligned words instead of precision/recall/F1 for all metrics.')
    parser.add_argument('--no-enhanced', dest='enhanced', action='store_false', default=True, help='Turn off evaluation of enhanced dependencies.')
    parser.add_argument('--enhancements', type=str, default='0', help='Level of enhancements in the gold data (see guidelines) 0=all (default), 1=no gapping, 2=no shared parents, 3=no shared dependents 4=no control, 5=no external arguments, 6=no lemma info, combinations: 12=both 1 and 2 apply, etc.')
    parser.add_argument('--no-empty-nodes', default=False, help='Empty nodes have been collapsed (needed to correctly evaluate enhanced/gapping). Raise exception if an empty node is encountered.')
    parser.add_argument('--multiple-roots-okay', default=False, action='store_true', help='A single sentence can have multiple nodes with HEAD=0.')
    args = parser.parse_args()
    evaluation = evaluate_wrapper(args)
    results = build_evaluation_table(evaluation, args.verbose, args.counts, args.enhanced)
    print(results)
if __name__ == '__main__':
    main()

class TestAlignment(unittest.TestCase):

    @staticmethod
    def _load_words(words):
        if False:
            print('Hello World!')
        'Prepare fake CoNLL-U files with fake HEAD to prevent multiple roots errors.'
        (lines, num_words) = ([], 0)
        for w in words:
            parts = w.split(' ')
            if len(parts) == 1:
                num_words += 1
                lines.append('{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_'.format(num_words, parts[0], int(num_words > 1)))
            else:
                lines.append('{}-{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_'.format(num_words + 1, num_words + len(parts) - 1, parts[0]))
                for part in parts[1:]:
                    num_words += 1
                    lines.append('{}\t{}\t_\t_\t_\t_\t{}\t_\t_\t_'.format(num_words, part, int(num_words > 1)))
        return load_conllu((io.StringIO if sys.version_info >= (3, 0) else io.BytesIO)('\n'.join(lines + ['\n'])))

    def _test_exception(self, gold, system):
        if False:
            i = 10
            return i + 15
        self.assertRaises(UDError, evaluate, self._load_words(gold), self._load_words(system))

    def _test_ok(self, gold, system, correct):
        if False:
            for i in range(10):
                print('nop')
        metrics = evaluate(self._load_words(gold), self._load_words(system))
        gold_words = sum((max(1, len(word.split(' ')) - 1) for word in gold))
        system_words = sum((max(1, len(word.split(' ')) - 1) for word in system))
        self.assertEqual((metrics['Words'].precision, metrics['Words'].recall, metrics['Words'].f1), (correct / system_words, correct / gold_words, 2 * correct / (gold_words + system_words)))

    def test_exception(self):
        if False:
            while True:
                i = 10
        self._test_exception(['a'], ['b'])

    def test_equal(self):
        if False:
            return 10
        self._test_ok(['a'], ['a'], 1)
        self._test_ok(['a', 'b', 'c'], ['a', 'b', 'c'], 3)

    def test_equal_with_multiword(self):
        if False:
            while True:
                i = 10
        self._test_ok(['abc a b c'], ['a', 'b', 'c'], 3)
        self._test_ok(['a', 'bc b c', 'd'], ['a', 'b', 'c', 'd'], 4)
        self._test_ok(['abcd a b c d'], ['ab a b', 'cd c d'], 4)
        self._test_ok(['abc a b c', 'de d e'], ['a', 'bcd b c d', 'e'], 5)

    def test_alignment(self):
        if False:
            return 10
        self._test_ok(['abcd'], ['a', 'b', 'c', 'd'], 0)
        self._test_ok(['abc', 'd'], ['a', 'b', 'c', 'd'], 1)
        self._test_ok(['a', 'bc', 'd'], ['a', 'b', 'c', 'd'], 2)
        self._test_ok(['a', 'bc b c', 'd'], ['a', 'b', 'cd'], 2)
        self._test_ok(['abc a BX c', 'def d EX f'], ['ab a b', 'cd c d', 'ef e f'], 4)
        self._test_ok(['ab a b', 'cd bc d'], ['a', 'bc', 'd'], 2)
        self._test_ok(['a', 'bc b c', 'd'], ['ab AX BX', 'cd CX a'], 1)