from nltk.corpus import wordnet

def lesk(context_sentence, ambiguous_word, pos=None, synsets=None, lang='eng'):
    if False:
        while True:
            i = 10
    'Return a synset for an ambiguous word in a context.\n\n    :param iter context_sentence: The context sentence where the ambiguous word\n         occurs, passed as an iterable of words.\n    :param str ambiguous_word: The ambiguous word that requires WSD.\n    :param str pos: A specified Part-of-Speech (POS).\n    :param iter synsets: Possible synsets of the ambiguous word.\n    :param str lang: WordNet language.\n    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.\n\n    This function is an implementation of the original Lesk algorithm (1986) [1].\n\n    Usage example::\n\n        >>> lesk([\'I\', \'went\', \'to\', \'the\', \'bank\', \'to\', \'deposit\', \'money\', \'.\'], \'bank\', \'n\')\n        Synset(\'savings_bank.n.02\')\n\n    [1] Lesk, Michael. "Automatic sense disambiguation using machine\n    readable dictionaries: how to tell a pine cone from an ice cream\n    cone." Proceedings of the 5th Annual International Conference on\n    Systems Documentation. ACM, 1986.\n    https://dl.acm.org/citation.cfm?id=318728\n    '
    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word, lang=lang)
    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]
    if not synsets:
        return None
    (_, sense) = max(((len(context.intersection(ss.definition().split())), ss) for ss in synsets))
    return sense