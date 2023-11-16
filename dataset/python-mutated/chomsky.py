"""
CHOMSKY is an aid to writing linguistic papers in the style
of the great master.  It is based on selected phrases taken
from actual books and articles written by Noam Chomsky.
Upon request, it assembles the phrases in the elegant
stylistic patterns that Chomsky is noted for.
To generate n sentences of linguistic wisdom, type

    (CHOMSKY n)  -- for example
    (CHOMSKY 5) generates half a screen of linguistic truth.
"""
leadins = 'To characterize a linguistic level L,\n    On the other hand,\n    This suggests that\n    It appears that\n    Furthermore,\n    We will bring evidence in favor of the following thesis:\n    To provide a constituent structure for T(Z,K),\n    From C1, it follows that\n    For any transformation which is sufficiently diversified in application to be of any interest,\n    Analogously,\n    Clearly,\n    Note that\n    Of course,\n    Suppose, for instance, that\n    Thus\n    With this clarification,\n    Conversely,\n    We have already seen that\n    By combining adjunctions and certain deformations,\n    I suggested that these results would follow from the assumption that\n    If the position of the trace in (99c) were only relatively inaccessible to movement,\n    However, this assumption is not correct, since\n    Comparing these examples with their parasitic gap counterparts in (96) and (97), we see that\n    In the discussion of resumptive pronouns following (81),\n    So far,\n    Nevertheless,\n    For one thing,\n    Summarizing, then, we assume that\n    A consequence of the approach just outlined is that\n    Presumably,\n    On our assumptions,\n    It may be, then, that\n    It must be emphasized, once again, that\n    Let us continue to suppose that\n    Notice, incidentally, that '
subjects = " the notion of level of grammaticalness\n    a case of semigrammaticalness of a different sort\n    most of the methodological work in modern linguistics\n    a subset of English sentences interesting on quite independent grounds\n    the natural general principle that will subsume this case\n    an important property of these three types of EC\n    any associated supporting element\n    the appearance of parasitic gaps in domains relatively inaccessible to ordinary extraction\n    the speaker-hearer's linguistic intuition\n    the descriptive power of the base component\n    the earlier discussion of deviance\n    this analysis of a formative as a pair of sets of features\n    this selectionally introduced contextual feature\n    a descriptively adequate grammar\n    the fundamental error of regarding functional notions as categorial\n    relational information\n    the systematic use of complex symbols\n    the theory of syntactic features developed earlier"
verbs = 'can be defined in such a way as to impose\n    delimits\n    suffices to account for\n    cannot be arbitrary in\n    is not subject to\n    does not readily tolerate\n    raises serious doubts about\n    is not quite equivalent to\n    does not affect the structure of\n    may remedy and, at the same time, eliminate\n    is not to be considered in determining\n    is to be regarded as\n    is unspecified with respect to\n    is, apparently, determined by\n    is necessary to impose an interpretation on\n    appears to correlate rather closely with\n    is rather different from'
objects = ' problems of phonemic and morphological analysis.\n    a corpus of utterance tokens upon which conformity has been defined by the paired utterance test.\n    the traditional practice of grammarians.\n    the levels of acceptability from fairly high (e.g. (99a)) to virtual gibberish (e.g. (98d)).\n    a stipulation to place the constructions into these various categories.\n    a descriptive fact.\n    a parasitic gap construction.\n    the extended c-command discussed in connection with (34).\n    the ultimate standard that determines the accuracy of any proposed grammar.\n    the system of base rules exclusive of the lexicon.\n    irrelevant intervening contexts in selectional rules.\n    nondistinctness in the sense of distinctive feature theory.\n    a general convention regarding the forms of the grammar.\n    an abstract underlying order.\n    an important distinction in language use.\n    the requirement that branching is not tolerated within the dominance scope of a complex symbol.\n    the strong generative capacity of the theory.'
import random
import textwrap
from itertools import chain, islice

def generate_chomsky(times=5, line_length=72):
    if False:
        for i in range(10):
            print('nop')
    parts = []
    for part in (leadins, subjects, verbs, objects):
        phraselist = list(map(str.strip, part.splitlines()))
        random.shuffle(phraselist)
        parts.append(phraselist)
    output = chain.from_iterable(islice(zip(*parts), 0, times))
    print(textwrap.fill(' '.join(output), line_length))
if __name__ == '__main__':
    generate_chomsky()