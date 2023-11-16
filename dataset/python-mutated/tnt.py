"""
Implementation of 'TnT - A Statisical Part of Speech Tagger'
by Thorsten Brants

https://aclanthology.org/A00-1031.pdf
"""
from math import log
from operator import itemgetter
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI

class TnT(TaggerI):
    """
    TnT - Statistical POS tagger

    IMPORTANT NOTES:

    * DOES NOT AUTOMATICALLY DEAL WITH UNSEEN WORDS

      - It is possible to provide an untrained POS tagger to
        create tags for unknown words, see __init__ function

    * SHOULD BE USED WITH SENTENCE-DELIMITED INPUT

      - Due to the nature of this tagger, it works best when
        trained over sentence delimited input.
      - However it still produces good results if the training
        data and testing data are separated on all punctuation eg: [,.?!]
      - Input for training is expected to be a list of sentences
        where each sentence is a list of (word, tag) tuples
      - Input for tag function is a single sentence
        Input for tagdata function is a list of sentences
        Output is of a similar form

    * Function provided to process text that is unsegmented

      - Please see basic_sent_chop()


    TnT uses a second order Markov model to produce tags for
    a sequence of input, specifically:

      argmax [Proj(P(t_i|t_i-1,t_i-2)P(w_i|t_i))] P(t_T+1 | t_T)

    IE: the maximum projection of a set of probabilities

    The set of possible tags for a given word is derived
    from the training data. It is the set of all tags
    that exact word has been assigned.

    To speed up and get more precision, we can use log addition
    to instead multiplication, specifically:

      argmax [Sigma(log(P(t_i|t_i-1,t_i-2))+log(P(w_i|t_i)))] +
             log(P(t_T+1|t_T))

    The probability of a tag for a given word is the linear
    interpolation of 3 markov models; a zero-order, first-order,
    and a second order model.

      P(t_i| t_i-1, t_i-2) = l1*P(t_i) + l2*P(t_i| t_i-1) +
                             l3*P(t_i| t_i-1, t_i-2)

    A beam search is used to limit the memory usage of the algorithm.
    The degree of the beam can be changed using N in the initialization.
    N represents the maximum number of possible solutions to maintain
    while tagging.

    It is possible to differentiate the tags which are assigned to
    capitalized words. However this does not result in a significant
    gain in the accuracy of the results.
    """

    def __init__(self, unk=None, Trained=False, N=1000, C=False):
        if False:
            i = 10
            return i + 15
        '\n        Construct a TnT statistical tagger. Tagger must be trained\n        before being used to tag input.\n\n        :param unk: instance of a POS tagger, conforms to TaggerI\n        :type  unk: TaggerI\n        :param Trained: Indication that the POS tagger is trained or not\n        :type  Trained: bool\n        :param N: Beam search degree (see above)\n        :type  N: int\n        :param C: Capitalization flag\n        :type  C: bool\n\n        Initializer, creates frequency distributions to be used\n        for tagging\n\n        _lx values represent the portion of the tri/bi/uni taggers\n        to be used to calculate the probability\n\n        N value is the number of possible solutions to maintain\n        while tagging. A good value for this is 1000\n\n        C is a boolean value which specifies to use or\n        not use the Capitalization of the word as additional\n        information for tagging.\n        NOTE: using capitalization may not increase the accuracy\n        of the tagger\n        '
        self._uni = FreqDist()
        self._bi = ConditionalFreqDist()
        self._tri = ConditionalFreqDist()
        self._wd = ConditionalFreqDist()
        self._eos = ConditionalFreqDist()
        self._l1 = 0.0
        self._l2 = 0.0
        self._l3 = 0.0
        self._N = N
        self._C = C
        self._T = Trained
        self._unk = unk
        self.unknown = 0
        self.known = 0

    def train(self, data):
        if False:
            print('Hello World!')
        '\n        Uses a set of tagged data to train the tagger.\n        If an unknown word tagger is specified,\n        it is trained on the same data.\n\n        :param data: List of lists of (word, tag) tuples\n        :type data: tuple(str)\n        '
        C = False
        if self._unk is not None and self._T == False:
            self._unk.train(data)
        for sent in data:
            history = [('BOS', False), ('BOS', False)]
            for (w, t) in sent:
                if self._C and w[0].isupper():
                    C = True
                self._wd[w][t] += 1
                self._uni[t, C] += 1
                self._bi[history[1]][t, C] += 1
                self._tri[tuple(history)][t, C] += 1
                history.append((t, C))
                history.pop(0)
                C = False
            self._eos[t]['EOS'] += 1
        self._compute_lambda()

    def _compute_lambda(self):
        if False:
            return 10
        '\n        creates lambda values based upon training data\n\n        NOTE: no need to explicitly reference C,\n        it is contained within the tag variable :: tag == (tag,C)\n\n        for each tag trigram (t1, t2, t3)\n        depending on the maximum value of\n        - f(t1,t2,t3)-1 / f(t1,t2)-1\n        - f(t2,t3)-1 / f(t2)-1\n        - f(t3)-1 / N-1\n\n        increment l3,l2, or l1 by f(t1,t2,t3)\n\n        ISSUES -- Resolutions:\n        if 2 values are equal, increment both lambda values\n        by (f(t1,t2,t3) / 2)\n        '
        tl1 = 0.0
        tl2 = 0.0
        tl3 = 0.0
        for history in self._tri.conditions():
            (h1, h2) = history
            for tag in self._tri[history].keys():
                if self._uni[tag] == 1:
                    continue
                c3 = self._safe_div(self._tri[history][tag] - 1, self._tri[history].N() - 1)
                c2 = self._safe_div(self._bi[h2][tag] - 1, self._bi[h2].N() - 1)
                c1 = self._safe_div(self._uni[tag] - 1, self._uni.N() - 1)
                if c1 > c3 and c1 > c2:
                    tl1 += self._tri[history][tag]
                elif c2 > c3 and c2 > c1:
                    tl2 += self._tri[history][tag]
                elif c3 > c2 and c3 > c1:
                    tl3 += self._tri[history][tag]
                elif c3 == c2 and c3 > c1:
                    tl2 += self._tri[history][tag] / 2.0
                    tl3 += self._tri[history][tag] / 2.0
                elif c2 == c1 and c1 > c3:
                    tl1 += self._tri[history][tag] / 2.0
                    tl2 += self._tri[history][tag] / 2.0
                else:
                    pass
        self._l1 = tl1 / (tl1 + tl2 + tl3)
        self._l2 = tl2 / (tl1 + tl2 + tl3)
        self._l3 = tl3 / (tl1 + tl2 + tl3)

    def _safe_div(self, v1, v2):
        if False:
            i = 10
            return i + 15
        '\n        Safe floating point division function, does not allow division by 0\n        returns -1 if the denominator is 0\n        '
        if v2 == 0:
            return -1
        else:
            return v1 / v2

    def tagdata(self, data):
        if False:
            while True:
                i = 10
        '\n        Tags each sentence in a list of sentences\n\n        :param data:list of list of words\n        :type data: [[string,],]\n        :return: list of list of (word, tag) tuples\n\n        Invokes tag(sent) function for each sentence\n        compiles the results into a list of tagged sentences\n        each tagged sentence is a list of (word, tag) tuples\n        '
        res = []
        for sent in data:
            res1 = self.tag(sent)
            res.append(res1)
        return res

    def tag(self, data):
        if False:
            while True:
                i = 10
        "\n        Tags a single sentence\n\n        :param data: list of words\n        :type data: [string,]\n\n        :return: [(word, tag),]\n\n        Calls recursive function '_tagword'\n        to produce a list of tags\n\n        Associates the sequence of returned tags\n        with the correct words in the input sequence\n\n        returns a list of (word, tag) tuples\n        "
        current_state = [(['BOS', 'BOS'], 0.0)]
        sent = list(data)
        tags = self._tagword(sent, current_state)
        res = []
        for i in range(len(sent)):
            (t, C) = tags[i + 2]
            res.append((sent[i], t))
        return res

    def _tagword(self, sent, current_states):
        if False:
            while True:
                i = 10
        '\n        :param sent : List of words remaining in the sentence\n        :type sent  : [word,]\n        :param current_states : List of possible tag combinations for\n                                the sentence so far, and the log probability\n                                associated with each tag combination\n        :type current_states  : [([tag, ], logprob), ]\n\n        Tags the first word in the sentence and\n        recursively tags the reminder of sentence\n\n        Uses formula specified above to calculate the probability\n        of a particular tag\n        '
        if sent == []:
            (h, logp) = current_states[0]
            return h
        word = sent[0]
        sent = sent[1:]
        new_states = []
        C = False
        if self._C and word[0].isupper():
            C = True
        if word in self._wd:
            self.known += 1
            for (history, curr_sent_logprob) in current_states:
                logprobs = []
                for t in self._wd[word].keys():
                    tC = (t, C)
                    p_uni = self._uni.freq(tC)
                    p_bi = self._bi[history[-1]].freq(tC)
                    p_tri = self._tri[tuple(history[-2:])].freq(tC)
                    p_wd = self._wd[word][t] / self._uni[tC]
                    p = self._l1 * p_uni + self._l2 * p_bi + self._l3 * p_tri
                    p2 = log(p, 2) + log(p_wd, 2)
                    new_states.append((history + [tC], curr_sent_logprob + p2))
        else:
            self.unknown += 1
            p = 1
            if self._unk is None:
                tag = ('Unk', C)
            else:
                [(_w, t)] = list(self._unk.tag([word]))
                tag = (t, C)
            for (history, logprob) in current_states:
                history.append(tag)
            new_states = current_states
        new_states.sort(reverse=True, key=itemgetter(1))
        if len(new_states) > self._N:
            new_states = new_states[:self._N]
        return self._tagword(sent, new_states)

def basic_sent_chop(data, raw=True):
    if False:
        return 10
    '\n    Basic method for tokenizing input into sentences\n    for this tagger:\n\n    :param data: list of tokens (words or (word, tag) tuples)\n    :type data: str or tuple(str, str)\n    :param raw: boolean flag marking the input data\n                as a list of words or a list of tagged words\n    :type raw: bool\n    :return: list of sentences\n             sentences are a list of tokens\n             tokens are the same as the input\n\n    Function takes a list of tokens and separates the tokens into lists\n    where each list represents a sentence fragment\n    This function can separate both tagged and raw sequences into\n    basic sentences.\n\n    Sentence markers are the set of [,.!?]\n\n    This is a simple method which enhances the performance of the TnT\n    tagger. Better sentence tokenization will further enhance the results.\n    '
    new_data = []
    curr_sent = []
    sent_mark = [',', '.', '?', '!']
    if raw:
        for word in data:
            if word in sent_mark:
                curr_sent.append(word)
                new_data.append(curr_sent)
                curr_sent = []
            else:
                curr_sent.append(word)
    else:
        for (word, tag) in data:
            if word in sent_mark:
                curr_sent.append((word, tag))
                new_data.append(curr_sent)
                curr_sent = []
            else:
                curr_sent.append((word, tag))
    return new_data

def demo():
    if False:
        return 10
    from nltk.corpus import brown
    sents = list(brown.tagged_sents())
    test = list(brown.sents())
    tagger = TnT()
    tagger.train(sents[200:1000])
    tagged_data = tagger.tagdata(test[100:120])
    for j in range(len(tagged_data)):
        s = tagged_data[j]
        t = sents[j + 100]
        for i in range(len(s)):
            print(s[i], '--', t[i])
        print()

def demo2():
    if False:
        for i in range(10):
            print('nop')
    from nltk.corpus import treebank
    d = list(treebank.tagged_sents())
    t = TnT(N=1000, C=False)
    s = TnT(N=1000, C=True)
    t.train(d[11 * 100:])
    s.train(d[11 * 100:])
    for i in range(10):
        tacc = t.accuracy(d[i * 100:(i + 1) * 100])
        tp_un = t.unknown / (t.known + t.unknown)
        tp_kn = t.known / (t.known + t.unknown)
        t.unknown = 0
        t.known = 0
        print('Capitalization off:')
        print('Accuracy:', tacc)
        print('Percentage known:', tp_kn)
        print('Percentage unknown:', tp_un)
        print('Accuracy over known words:', tacc / tp_kn)
        sacc = s.accuracy(d[i * 100:(i + 1) * 100])
        sp_un = s.unknown / (s.known + s.unknown)
        sp_kn = s.known / (s.known + s.unknown)
        s.unknown = 0
        s.known = 0
        print('Capitalization on:')
        print('Accuracy:', sacc)
        print('Percentage known:', sp_kn)
        print('Percentage unknown:', sp_un)
        print('Accuracy over known words:', sacc / sp_kn)

def demo3():
    if False:
        return 10
    from nltk.corpus import brown, treebank
    d = list(treebank.tagged_sents())
    e = list(brown.tagged_sents())
    d = d[:1000]
    e = e[:1000]
    d10 = int(len(d) * 0.1)
    e10 = int(len(e) * 0.1)
    tknacc = 0
    sknacc = 0
    tallacc = 0
    sallacc = 0
    tknown = 0
    sknown = 0
    for i in range(10):
        t = TnT(N=1000, C=False)
        s = TnT(N=1000, C=False)
        dtest = d[i * d10:(i + 1) * d10]
        etest = e[i * e10:(i + 1) * e10]
        dtrain = d[:i * d10] + d[(i + 1) * d10:]
        etrain = e[:i * e10] + e[(i + 1) * e10:]
        t.train(dtrain)
        s.train(etrain)
        tacc = t.accuracy(dtest)
        tp_un = t.unknown / (t.known + t.unknown)
        tp_kn = t.known / (t.known + t.unknown)
        tknown += tp_kn
        t.unknown = 0
        t.known = 0
        sacc = s.accuracy(etest)
        sp_un = s.unknown / (s.known + s.unknown)
        sp_kn = s.known / (s.known + s.unknown)
        sknown += sp_kn
        s.unknown = 0
        s.known = 0
        tknacc += tacc / tp_kn
        sknacc += sacc / tp_kn
        tallacc += tacc
        sallacc += sacc
    print('brown: acc over words known:', 10 * tknacc)
    print('     : overall accuracy:', 10 * tallacc)
    print('     : words known:', 10 * tknown)
    print('treebank: acc over words known:', 10 * sknacc)
    print('        : overall accuracy:', 10 * sallacc)
    print('        : words known:', 10 * sknown)