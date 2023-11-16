"""Counts Paice's performance statistics for evaluating stemming algorithms.

What is required:
 - A dictionary of words grouped by their real lemmas
 - A dictionary of words grouped by stems from a stemming algorithm

When these are given, Understemming Index (UI), Overstemming Index (OI),
Stemming Weight (SW) and Error-rate relative to truncation (ERRT) are counted.

References:
Chris D. Paice (1994). An evaluation method for stemming algorithms.
In Proceedings of SIGIR, 42--50.
"""
from math import sqrt

def get_words_from_dictionary(lemmas):
    if False:
        print('Hello World!')
    '\n    Get original set of words used for analysis.\n\n    :param lemmas: A dictionary where keys are lemmas and values are sets\n        or lists of words corresponding to that lemma.\n    :type lemmas: dict(str): list(str)\n    :return: Set of words that exist as values in the dictionary\n    :rtype: set(str)\n    '
    words = set()
    for lemma in lemmas:
        words.update(set(lemmas[lemma]))
    return words

def _truncate(words, cutlength):
    if False:
        i = 10
        return i + 15
    'Group words by stems defined by truncating them at given length.\n\n    :param words: Set of words used for analysis\n    :param cutlength: Words are stemmed by cutting at this length.\n    :type words: set(str) or list(str)\n    :type cutlength: int\n    :return: Dictionary where keys are stems and values are sets of words\n    corresponding to that stem.\n    :rtype: dict(str): set(str)\n    '
    stems = {}
    for word in words:
        stem = word[:cutlength]
        try:
            stems[stem].update([word])
        except KeyError:
            stems[stem] = {word}
    return stems

def _count_intersection(l1, l2):
    if False:
        i = 10
        return i + 15
    'Count intersection between two line segments defined by coordinate pairs.\n\n    :param l1: Tuple of two coordinate pairs defining the first line segment\n    :param l2: Tuple of two coordinate pairs defining the second line segment\n    :type l1: tuple(float, float)\n    :type l2: tuple(float, float)\n    :return: Coordinates of the intersection\n    :rtype: tuple(float, float)\n    '
    (x1, y1) = l1[0]
    (x2, y2) = l1[1]
    (x3, y3) = l2[0]
    (x4, y4) = l2[1]
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0.0:
        if x1 == x2 == x3 == x4 == 0.0:
            return (0.0, y4)
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (x, y)

def _get_derivative(coordinates):
    if False:
        while True:
            i = 10
    'Get derivative of the line from (0,0) to given coordinates.\n\n    :param coordinates: A coordinate pair\n    :type coordinates: tuple(float, float)\n    :return: Derivative; inf if x is zero\n    :rtype: float\n    '
    try:
        return coordinates[1] / coordinates[0]
    except ZeroDivisionError:
        return float('inf')

def _calculate_cut(lemmawords, stems):
    if False:
        for i in range(10):
            print('nop')
    'Count understemmed and overstemmed pairs for (lemma, stem) pair with common words.\n\n    :param lemmawords: Set or list of words corresponding to certain lemma.\n    :param stems: A dictionary where keys are stems and values are sets\n    or lists of words corresponding to that stem.\n    :type lemmawords: set(str) or list(str)\n    :type stems: dict(str): set(str)\n    :return: Amount of understemmed and overstemmed pairs contributed by words\n    existing in both lemmawords and stems.\n    :rtype: tuple(float, float)\n    '
    (umt, wmt) = (0.0, 0.0)
    for stem in stems:
        cut = set(lemmawords) & set(stems[stem])
        if cut:
            cutcount = len(cut)
            stemcount = len(stems[stem])
            umt += cutcount * (len(lemmawords) - cutcount)
            wmt += cutcount * (stemcount - cutcount)
    return (umt, wmt)

def _calculate(lemmas, stems):
    if False:
        print('Hello World!')
    'Calculate actual and maximum possible amounts of understemmed and overstemmed word pairs.\n\n    :param lemmas: A dictionary where keys are lemmas and values are sets\n    or lists of words corresponding to that lemma.\n    :param stems: A dictionary where keys are stems and values are sets\n    or lists of words corresponding to that stem.\n    :type lemmas: dict(str): list(str)\n    :type stems: dict(str): set(str)\n    :return: Global unachieved merge total (gumt),\n    global desired merge total (gdmt),\n    global wrongly merged total (gwmt) and\n    global desired non-merge total (gdnt).\n    :rtype: tuple(float, float, float, float)\n    '
    n = sum((len(lemmas[word]) for word in lemmas))
    (gdmt, gdnt, gumt, gwmt) = (0.0, 0.0, 0.0, 0.0)
    for lemma in lemmas:
        lemmacount = len(lemmas[lemma])
        gdmt += lemmacount * (lemmacount - 1)
        gdnt += lemmacount * (n - lemmacount)
        (umt, wmt) = _calculate_cut(lemmas[lemma], stems)
        gumt += umt
        gwmt += wmt
    return (gumt / 2, gdmt / 2, gwmt / 2, gdnt / 2)

def _indexes(gumt, gdmt, gwmt, gdnt):
    if False:
        while True:
            i = 10
    'Count Understemming Index (UI), Overstemming Index (OI) and Stemming Weight (SW).\n\n    :param gumt, gdmt, gwmt, gdnt: Global unachieved merge total (gumt),\n    global desired merge total (gdmt),\n    global wrongly merged total (gwmt) and\n    global desired non-merge total (gdnt).\n    :type gumt, gdmt, gwmt, gdnt: float\n    :return: Understemming Index (UI),\n    Overstemming Index (OI) and\n    Stemming Weight (SW).\n    :rtype: tuple(float, float, float)\n    '
    try:
        ui = gumt / gdmt
    except ZeroDivisionError:
        ui = 0.0
    try:
        oi = gwmt / gdnt
    except ZeroDivisionError:
        oi = 0.0
    try:
        sw = oi / ui
    except ZeroDivisionError:
        if oi == 0.0:
            sw = float('nan')
        else:
            sw = float('inf')
    return (ui, oi, sw)

class Paice:
    """Class for storing lemmas, stems and evaluation metrics."""

    def __init__(self, lemmas, stems):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param lemmas: A dictionary where keys are lemmas and values are sets\n            or lists of words corresponding to that lemma.\n        :param stems: A dictionary where keys are stems and values are sets\n            or lists of words corresponding to that stem.\n        :type lemmas: dict(str): list(str)\n        :type stems: dict(str): set(str)\n        '
        self.lemmas = lemmas
        self.stems = stems
        self.coords = []
        (self.gumt, self.gdmt, self.gwmt, self.gdnt) = (None, None, None, None)
        (self.ui, self.oi, self.sw) = (None, None, None)
        self.errt = None
        self.update()

    def __str__(self):
        if False:
            return 10
        text = ['Global Unachieved Merge Total (GUMT): %s\n' % self.gumt]
        text.append('Global Desired Merge Total (GDMT): %s\n' % self.gdmt)
        text.append('Global Wrongly-Merged Total (GWMT): %s\n' % self.gwmt)
        text.append('Global Desired Non-merge Total (GDNT): %s\n' % self.gdnt)
        text.append('Understemming Index (GUMT / GDMT): %s\n' % self.ui)
        text.append('Overstemming Index (GWMT / GDNT): %s\n' % self.oi)
        text.append('Stemming Weight (OI / UI): %s\n' % self.sw)
        text.append('Error-Rate Relative to Truncation (ERRT): %s\r\n' % self.errt)
        coordinates = ' '.join(['(%s, %s)' % item for item in self.coords])
        text.append('Truncation line: %s' % coordinates)
        return ''.join(text)

    def _get_truncation_indexes(self, words, cutlength):
        if False:
            return 10
        "Count (UI, OI) when stemming is done by truncating words at 'cutlength'.\n\n        :param words: Words used for the analysis\n        :param cutlength: Words are stemmed by cutting them at this length\n        :type words: set(str) or list(str)\n        :type cutlength: int\n        :return: Understemming and overstemming indexes\n        :rtype: tuple(int, int)\n        "
        truncated = _truncate(words, cutlength)
        (gumt, gdmt, gwmt, gdnt) = _calculate(self.lemmas, truncated)
        (ui, oi) = _indexes(gumt, gdmt, gwmt, gdnt)[:2]
        return (ui, oi)

    def _get_truncation_coordinates(self, cutlength=0):
        if False:
            i = 10
            return i + 15
        'Count (UI, OI) pairs for truncation points until we find the segment where (ui, oi) crosses the truncation line.\n\n        :param cutlength: Optional parameter to start counting from (ui, oi)\n        coordinates gotten by stemming at this length. Useful for speeding up\n        the calculations when you know the approximate location of the\n        intersection.\n        :type cutlength: int\n        :return: List of coordinate pairs that define the truncation line\n        :rtype: list(tuple(float, float))\n        '
        words = get_words_from_dictionary(self.lemmas)
        maxlength = max((len(word) for word in words))
        coords = []
        while cutlength <= maxlength:
            pair = self._get_truncation_indexes(words, cutlength)
            if pair not in coords:
                coords.append(pair)
            if pair == (0.0, 0.0):
                return coords
            if len(coords) >= 2 and pair[0] > 0.0:
                derivative1 = _get_derivative(coords[-2])
                derivative2 = _get_derivative(coords[-1])
                if derivative1 >= self.sw >= derivative2:
                    return coords
            cutlength += 1
        return coords

    def _errt(self):
        if False:
            for i in range(10):
                print('nop')
        'Count Error-Rate Relative to Truncation (ERRT).\n\n        :return: ERRT, length of the line from origo to (UI, OI) divided by\n        the length of the line from origo to the point defined by the same\n        line when extended until the truncation line.\n        :rtype: float\n        '
        self.coords = self._get_truncation_coordinates()
        if (0.0, 0.0) in self.coords:
            if (self.ui, self.oi) != (0.0, 0.0):
                return float('inf')
            else:
                return float('nan')
        if (self.ui, self.oi) == (0.0, 0.0):
            return 0.0
        intersection = _count_intersection(((0, 0), (self.ui, self.oi)), self.coords[-2:])
        op = sqrt(self.ui ** 2 + self.oi ** 2)
        ot = sqrt(intersection[0] ** 2 + intersection[1] ** 2)
        return op / ot

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        'Update statistics after lemmas and stems have been set.'
        (self.gumt, self.gdmt, self.gwmt, self.gdnt) = _calculate(self.lemmas, self.stems)
        (self.ui, self.oi, self.sw) = _indexes(self.gumt, self.gdmt, self.gwmt, self.gdnt)
        self.errt = self._errt()

def demo():
    if False:
        for i in range(10):
            print('nop')
    'Demonstration of the module.'
    lemmas = {'kneel': ['kneel', 'knelt'], 'range': ['range', 'ranged'], 'ring': ['ring', 'rang', 'rung']}
    stems = {'kneel': ['kneel'], 'knelt': ['knelt'], 'rang': ['rang', 'range', 'ranged'], 'ring': ['ring'], 'rung': ['rung']}
    print('Words grouped by their lemmas:')
    for lemma in sorted(lemmas):
        print('{} => {}'.format(lemma, ' '.join(lemmas[lemma])))
    print()
    print('Same words grouped by a stemming algorithm:')
    for stem in sorted(stems):
        print('{} => {}'.format(stem, ' '.join(stems[stem])))
    print()
    p = Paice(lemmas, stems)
    print(p)
    print()
    stems = {'kneel': ['kneel'], 'knelt': ['knelt'], 'rang': ['rang'], 'range': ['range', 'ranged'], 'ring': ['ring'], 'rung': ['rung']}
    print('Counting stats after changing stemming results:')
    for stem in sorted(stems):
        print('{} => {}'.format(stem, ' '.join(stems[stem])))
    print()
    p.stems = stems
    p.update()
    print(p)
if __name__ == '__main__':
    demo()