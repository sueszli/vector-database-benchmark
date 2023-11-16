from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template

@jsontags.register_tag
class Word(Feature):
    """
    Feature which examines the text (word) of nearby tokens.
    """
    json_tag = 'nltk.tag.brill.Word'

    @staticmethod
    def extract_property(tokens, index):
        if False:
            i = 10
            return i + 15
        "@return: The given token's text."
        return tokens[index][0]

@jsontags.register_tag
class Pos(Feature):
    """
    Feature which examines the tags of nearby tokens.
    """
    json_tag = 'nltk.tag.brill.Pos'

    @staticmethod
    def extract_property(tokens, index):
        if False:
            while True:
                i = 10
        "@return: The given token's tag."
        return tokens[index][1]

def nltkdemo18():
    if False:
        i = 10
        return i + 15
    '\n    Return 18 templates, from the original nltk demo, in multi-feature syntax\n    '
    return [Template(Pos([-1])), Template(Pos([1])), Template(Pos([-2])), Template(Pos([2])), Template(Pos([-2, -1])), Template(Pos([1, 2])), Template(Pos([-3, -2, -1])), Template(Pos([1, 2, 3])), Template(Pos([-1]), Pos([1])), Template(Word([-1])), Template(Word([1])), Template(Word([-2])), Template(Word([2])), Template(Word([-2, -1])), Template(Word([1, 2])), Template(Word([-3, -2, -1])), Template(Word([1, 2, 3])), Template(Word([-1]), Word([1]))]

def nltkdemo18plus():
    if False:
        i = 10
        return i + 15
    '\n    Return 18 templates, from the original nltk demo, and additionally a few\n    multi-feature ones (the motivation is easy comparison with nltkdemo18)\n    '
    return nltkdemo18() + [Template(Word([-1]), Pos([1])), Template(Pos([-1]), Word([1])), Template(Word([-1]), Word([0]), Pos([1])), Template(Pos([-1]), Word([0]), Word([1])), Template(Pos([-1]), Word([0]), Pos([1]))]

def fntbl37():
    if False:
        i = 10
        return i + 15
    '\n    Return 37 templates taken from the postagging task of the\n    fntbl distribution https://www.cs.jhu.edu/~rflorian/fntbl/\n    (37 is after excluding a handful which do not condition on Pos[0];\n    fntbl can do that but the current nltk implementation cannot.)\n    '
    return [Template(Word([0]), Word([1]), Word([2])), Template(Word([-1]), Word([0]), Word([1])), Template(Word([0]), Word([-1])), Template(Word([0]), Word([1])), Template(Word([0]), Word([2])), Template(Word([0]), Word([-2])), Template(Word([1, 2])), Template(Word([-2, -1])), Template(Word([1, 2, 3])), Template(Word([-3, -2, -1])), Template(Word([0]), Pos([2])), Template(Word([0]), Pos([-2])), Template(Word([0]), Pos([1])), Template(Word([0]), Pos([-1])), Template(Word([0])), Template(Word([-2])), Template(Word([2])), Template(Word([1])), Template(Word([-1])), Template(Pos([-1]), Pos([1])), Template(Pos([1]), Pos([2])), Template(Pos([-1]), Pos([-2])), Template(Pos([1])), Template(Pos([-1])), Template(Pos([-2])), Template(Pos([2])), Template(Pos([1, 2, 3])), Template(Pos([1, 2])), Template(Pos([-3, -2, -1])), Template(Pos([-2, -1])), Template(Pos([1]), Word([0]), Word([1])), Template(Pos([1]), Word([0]), Word([-1])), Template(Pos([-1]), Word([-1]), Word([0])), Template(Pos([-1]), Word([0]), Word([1])), Template(Pos([-2]), Pos([-1])), Template(Pos([1]), Pos([2])), Template(Pos([1]), Pos([2]), Word([1]))]

def brill24():
    if False:
        while True:
            i = 10
    '\n    Return 24 templates of the seminal TBL paper, Brill (1995)\n    '
    return [Template(Pos([-1])), Template(Pos([1])), Template(Pos([-2])), Template(Pos([2])), Template(Pos([-2, -1])), Template(Pos([1, 2])), Template(Pos([-3, -2, -1])), Template(Pos([1, 2, 3])), Template(Pos([-1]), Pos([1])), Template(Pos([-2]), Pos([-1])), Template(Pos([1]), Pos([2])), Template(Word([-1])), Template(Word([1])), Template(Word([-2])), Template(Word([2])), Template(Word([-2, -1])), Template(Word([1, 2])), Template(Word([-1, 0])), Template(Word([0, 1])), Template(Word([0])), Template(Word([-1]), Pos([-1])), Template(Word([1]), Pos([1])), Template(Word([0]), Word([-1]), Pos([-1])), Template(Word([0]), Word([1]), Pos([1]))]

def describe_template_sets():
    if False:
        while True:
            i = 10
    '\n    Print the available template sets in this demo, with a short description"\n    '
    import inspect
    import sys
    templatesets = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    for (name, obj) in templatesets:
        if name == 'describe_template_sets':
            continue
        print(name, obj.__doc__, '\n')

@jsontags.register_tag
class BrillTagger(TaggerI):
    """
    Brill's transformational rule-based tagger.  Brill taggers use an
    initial tagger (such as ``tag.DefaultTagger``) to assign an initial
    tag sequence to a text; and then apply an ordered list of
    transformational rules to correct the tags of individual tokens.
    These transformation rules are specified by the ``TagRule``
    interface.

    Brill taggers can be created directly, from an initial tagger and
    a list of transformational rules; but more often, Brill taggers
    are created by learning rules from a training corpus, using one
    of the TaggerTrainers available.
    """
    json_tag = 'nltk.tag.BrillTagger'

    def __init__(self, initial_tagger, rules, training_stats=None):
        if False:
            while True:
                i = 10
        '\n        :param initial_tagger: The initial tagger\n        :type initial_tagger: TaggerI\n\n        :param rules: An ordered list of transformation rules that\n            should be used to correct the initial tagging.\n        :type rules: list(TagRule)\n\n        :param training_stats: A dictionary of statistics collected\n            during training, for possible later use\n        :type training_stats: dict\n\n        '
        self._initial_tagger = initial_tagger
        self._rules = tuple(rules)
        self._training_stats = training_stats

    def encode_json_obj(self):
        if False:
            for i in range(10):
                print('nop')
        return (self._initial_tagger, self._rules, self._training_stats)

    @classmethod
    def decode_json_obj(cls, obj):
        if False:
            return 10
        (_initial_tagger, _rules, _training_stats) = obj
        return cls(_initial_tagger, _rules, _training_stats)

    def rules(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the ordered list of  transformation rules that this tagger has learnt\n\n        :return: the ordered list of transformation rules that correct the initial tagging\n        :rtype: list of Rules\n        '
        return self._rules

    def train_stats(self, statistic=None):
        if False:
            return 10
        '\n        Return a named statistic collected during training, or a dictionary of all\n        available statistics if no name given\n\n        :param statistic: name of statistic\n        :type statistic: str\n        :return: some statistic collected during training of this tagger\n        :rtype: any (but usually a number)\n        '
        if statistic is None:
            return self._training_stats
        else:
            return self._training_stats.get(statistic)

    def tag(self, tokens):
        if False:
            print('Hello World!')
        tagged_tokens = self._initial_tagger.tag(tokens)
        tag_to_positions = defaultdict(set)
        for (i, (token, tag)) in enumerate(tagged_tokens):
            tag_to_positions[tag].add(i)
        for rule in self._rules:
            positions = tag_to_positions.get(rule.original_tag, [])
            changed = rule.apply(tagged_tokens, positions)
            for i in changed:
                tag_to_positions[rule.original_tag].remove(i)
                tag_to_positions[rule.replacement_tag].add(i)
        return tagged_tokens

    def print_template_statistics(self, test_stats=None, printunused=True):
        if False:
            i = 10
            return i + 15
        '\n        Print a list of all templates, ranked according to efficiency.\n\n        If test_stats is available, the templates are ranked according to their\n        relative contribution (summed for all rules created from a given template,\n        weighted by score) to the performance on the test set. If no test_stats, then\n        statistics collected during training are used instead. There is also\n        an unweighted measure (just counting the rules). This is less informative,\n        though, as many low-score rules will appear towards end of training.\n\n        :param test_stats: dictionary of statistics collected during testing\n        :type test_stats: dict of str -> any (but usually numbers)\n        :param printunused: if True, print a list of all unused templates\n        :type printunused: bool\n        :return: None\n        :rtype: None\n        '
        tids = [r.templateid for r in self._rules]
        train_stats = self.train_stats()
        trainscores = train_stats['rulescores']
        assert len(trainscores) == len(tids), 'corrupt statistics: {} train scores for {} rules'.format(trainscores, tids)
        template_counts = Counter(tids)
        weighted_traincounts = Counter()
        for (tid, score) in zip(tids, trainscores):
            weighted_traincounts[tid] += score
        tottrainscores = sum(trainscores)

        def det_tplsort(tpl_value):
            if False:
                while True:
                    i = 10
            return (tpl_value[1], repr(tpl_value[0]))

        def print_train_stats():
            if False:
                print('Hello World!')
            print('TEMPLATE STATISTICS (TRAIN)  {} templates, {} rules)'.format(len(template_counts), len(tids)))
            print('TRAIN ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} final: {finalerrors:5d} {finalacc:.4f}'.format(**train_stats))
            head = '#ID | Score (train) |  #Rules     | Template'
            print(head, '\n', '-' * len(head), sep='')
            train_tplscores = sorted(weighted_traincounts.items(), key=det_tplsort, reverse=True)
            for (tid, trainscore) in train_tplscores:
                s = '{} | {:5d}   {:5.3f} |{:4d}   {:.3f} | {}'.format(tid, trainscore, trainscore / tottrainscores, template_counts[tid], template_counts[tid] / len(tids), Template.ALLTEMPLATES[int(tid)])
                print(s)

        def print_testtrain_stats():
            if False:
                print('Hello World!')
            testscores = test_stats['rulescores']
            print('TEMPLATE STATISTICS (TEST AND TRAIN) ({} templates, {} rules)'.format(len(template_counts), len(tids)))
            print('TEST  ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} final: {finalerrors:5d} {finalacc:.4f} '.format(**test_stats))
            print('TRAIN ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} final: {finalerrors:5d} {finalacc:.4f} '.format(**train_stats))
            weighted_testcounts = Counter()
            for (tid, score) in zip(tids, testscores):
                weighted_testcounts[tid] += score
            tottestscores = sum(testscores)
            head = '#ID | Score (test) | Score (train) |  #Rules     | Template'
            print(head, '\n', '-' * len(head), sep='')
            test_tplscores = sorted(weighted_testcounts.items(), key=det_tplsort, reverse=True)
            for (tid, testscore) in test_tplscores:
                s = '{:s} |{:5d}  {:6.3f} |  {:4d}   {:.3f} |{:4d}   {:.3f} | {:s}'.format(tid, testscore, testscore / tottestscores, weighted_traincounts[tid], weighted_traincounts[tid] / tottrainscores, template_counts[tid], template_counts[tid] / len(tids), Template.ALLTEMPLATES[int(tid)])
                print(s)

        def print_unused_templates():
            if False:
                return 10
            usedtpls = {int(tid) for tid in tids}
            unused = [(tid, tpl) for (tid, tpl) in enumerate(Template.ALLTEMPLATES) if tid not in usedtpls]
            print(f'UNUSED TEMPLATES ({len(unused)})')
            for (tid, tpl) in unused:
                print(f'{tid:03d} {str(tpl):s}')
        if test_stats is None:
            print_train_stats()
        else:
            print_testtrain_stats()
        print()
        if printunused:
            print_unused_templates()
        print()

    def batch_tag_incremental(self, sequences, gold):
        if False:
            i = 10
            return i + 15
        '\n        Tags by applying each rule to the entire corpus (rather than all rules to a\n        single sequence). The point is to collect statistics on the test set for\n        individual rules.\n\n        NOTE: This is inefficient (does not build any index, so will traverse the entire\n        corpus N times for N rules) -- usually you would not care about statistics for\n        individual rules and thus use batch_tag() instead\n\n        :param sequences: lists of token sequences (sentences, in some applications) to be tagged\n        :type sequences: list of list of strings\n        :param gold: the gold standard\n        :type gold: list of list of strings\n        :returns: tuple of (tagged_sequences, ordered list of rule scores (one for each rule))\n        '

        def counterrors(xs):
            if False:
                for i in range(10):
                    print('nop')
            return sum((t[1] != g[1] for pair in zip(xs, gold) for (t, g) in zip(*pair)))
        testing_stats = {}
        testing_stats['tokencount'] = sum((len(t) for t in sequences))
        testing_stats['sequencecount'] = len(sequences)
        tagged_tokenses = [self._initial_tagger.tag(tokens) for tokens in sequences]
        testing_stats['initialerrors'] = counterrors(tagged_tokenses)
        testing_stats['initialacc'] = 1 - testing_stats['initialerrors'] / testing_stats['tokencount']
        errors = [testing_stats['initialerrors']]
        for rule in self._rules:
            for tagged_tokens in tagged_tokenses:
                rule.apply(tagged_tokens)
            errors.append(counterrors(tagged_tokenses))
        testing_stats['rulescores'] = [err0 - err1 for (err0, err1) in zip(errors, errors[1:])]
        testing_stats['finalerrors'] = errors[-1]
        testing_stats['finalacc'] = 1 - testing_stats['finalerrors'] / testing_stats['tokencount']
        return (tagged_tokenses, testing_stats)