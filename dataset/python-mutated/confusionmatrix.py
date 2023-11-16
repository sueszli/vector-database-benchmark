from nltk.probability import FreqDist

class ConfusionMatrix:
    """
    The confusion matrix between a list of reference values and a
    corresponding list of test values.  Entry *[r,t]* of this
    matrix is a count of the number of times that the reference value
    *r* corresponds to the test value *t*.  E.g.:

        >>> from nltk.metrics import ConfusionMatrix
        >>> ref  = 'DET NN VB DET JJ NN NN IN DET NN'.split()
        >>> test = 'DET VB VB DET NN NN NN IN DET NN'.split()
        >>> cm = ConfusionMatrix(ref, test)
        >>> print(cm['NN', 'NN'])
        3

    Note that the diagonal entries *Ri=Tj* of this matrix
    corresponds to correct values; and the off-diagonal entries
    correspond to incorrect values.
    """

    def __init__(self, reference, test, sort_by_count=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new confusion matrix from a list of reference\n        values and a corresponding list of test values.\n\n        :type reference: list\n        :param reference: An ordered list of reference values.\n        :type test: list\n        :param test: A list of values to compare against the\n            corresponding reference values.\n        :raise ValueError: If ``reference`` and ``length`` do not have\n            the same length.\n        '
        if len(reference) != len(test):
            raise ValueError('Lists must have the same length.')
        if sort_by_count:
            ref_fdist = FreqDist(reference)
            test_fdist = FreqDist(test)

            def key(v):
                if False:
                    while True:
                        i = 10
                return -(ref_fdist[v] + test_fdist[v])
            values = sorted(set(reference + test), key=key)
        else:
            values = sorted(set(reference + test))
        indices = {val: i for (i, val) in enumerate(values)}
        confusion = [[0 for _ in values] for _ in values]
        max_conf = 0
        for (w, g) in zip(reference, test):
            confusion[indices[w]][indices[g]] += 1
            max_conf = max(max_conf, confusion[indices[w]][indices[g]])
        self._values = values
        self._indices = indices
        self._confusion = confusion
        self._max_conf = max_conf
        self._total = len(reference)
        self._correct = sum((confusion[i][i] for i in range(len(values))))

    def __getitem__(self, li_lj_tuple):
        if False:
            while True:
                i = 10
        '\n        :return: The number of times that value ``li`` was expected and\n        value ``lj`` was given.\n        :rtype: int\n        '
        (li, lj) = li_lj_tuple
        i = self._indices[li]
        j = self._indices[lj]
        return self._confusion[i][j]

    def __repr__(self):
        if False:
            return 10
        return f'<ConfusionMatrix: {self._correct}/{self._total} correct>'

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.pretty_format()

    def pretty_format(self, show_percents=False, values_in_chart=True, truncate=None, sort_by_count=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        :return: A multi-line string representation of this confusion matrix.\n        :type truncate: int\n        :param truncate: If specified, then only show the specified\n            number of values.  Any sorting (e.g., sort_by_count)\n            will be performed before truncation.\n        :param sort_by_count: If true, then sort by the count of each\n            label in the reference data.  I.e., labels that occur more\n            frequently in the reference label will be towards the left\n            edge of the matrix, and labels that occur less frequently\n            will be towards the right edge.\n\n        @todo: add marginals?\n        '
        confusion = self._confusion
        values = self._values
        if sort_by_count:
            values = sorted(values, key=lambda v: -sum(self._confusion[self._indices[v]]))
        if truncate:
            values = values[:truncate]
        if values_in_chart:
            value_strings = ['%s' % val for val in values]
        else:
            value_strings = [str(n + 1) for n in range(len(values))]
        valuelen = max((len(val) for val in value_strings))
        value_format = '%' + repr(valuelen) + 's | '
        if show_percents:
            entrylen = 6
            entry_format = '%5.1f%%'
            zerostr = '     .'
        else:
            entrylen = len(repr(self._max_conf))
            entry_format = '%' + repr(entrylen) + 'd'
            zerostr = ' ' * (entrylen - 1) + '.'
        s = ''
        for i in range(valuelen):
            s += ' ' * valuelen + ' |'
            for val in value_strings:
                if i >= valuelen - len(val):
                    s += val[i - valuelen + len(val)].rjust(entrylen + 1)
                else:
                    s += ' ' * (entrylen + 1)
            s += ' |\n'
        s += '{}-+-{}+\n'.format('-' * valuelen, '-' * ((entrylen + 1) * len(values)))
        for (val, li) in zip(value_strings, values):
            i = self._indices[li]
            s += value_format % val
            for lj in values:
                j = self._indices[lj]
                if confusion[i][j] == 0:
                    s += zerostr
                elif show_percents:
                    s += entry_format % (100.0 * confusion[i][j] / self._total)
                else:
                    s += entry_format % confusion[i][j]
                if i == j:
                    prevspace = s.rfind(' ')
                    s = s[:prevspace] + '<' + s[prevspace + 1:] + '>'
                else:
                    s += ' '
            s += '|\n'
        s += '{}-+-{}+\n'.format('-' * valuelen, '-' * ((entrylen + 1) * len(values)))
        s += '(row = reference; col = test)\n'
        if not values_in_chart:
            s += 'Value key:\n'
            for (i, value) in enumerate(values):
                s += '%6d: %s\n' % (i + 1, value)
        return s

    def key(self):
        if False:
            while True:
                i = 10
        values = self._values
        str = 'Value key:\n'
        indexlen = len(repr(len(values) - 1))
        key_format = '  %' + repr(indexlen) + 'd: %s\n'
        for i in range(len(values)):
            str += key_format % (i, values[i])
        return str

    def recall(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Given a value in the confusion matrix, return the recall\n        that corresponds to this value. The recall is defined as:\n\n        - *r* = true positive / (true positive + false positive)\n\n        and can loosely be considered the ratio of how often ``value``\n        was predicted correctly relative to how often ``value`` was\n        the true result.\n\n        :param value: value used in the ConfusionMatrix\n        :return: the recall corresponding to ``value``.\n        :rtype: float\n        '
        TP = self[value, value]
        TP_FN = sum((self[value, pred_value] for pred_value in self._values))
        if TP_FN == 0:
            return 0.0
        return TP / TP_FN

    def precision(self, value):
        if False:
            print('Hello World!')
        'Given a value in the confusion matrix, return the precision\n        that corresponds to this value. The precision is defined as:\n\n        - *p* = true positive / (true positive + false negative)\n\n        and can loosely be considered the ratio of how often ``value``\n        was predicted correctly relative to the number of predictions\n        for ``value``.\n\n        :param value: value used in the ConfusionMatrix\n        :return: the precision corresponding to ``value``.\n        :rtype: float\n        '
        TP = self[value, value]
        TP_FP = sum((self[real_value, value] for real_value in self._values))
        if TP_FP == 0:
            return 0.0
        return TP / TP_FP

    def f_measure(self, value, alpha=0.5):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a value used in the confusion matrix, return the f-measure\n        that corresponds to this value. The f-measure is the harmonic mean\n        of the ``precision`` and ``recall``, weighted by ``alpha``.\n        In particular, given the precision *p* and recall *r* defined by:\n\n        - *p* = true positive / (true positive + false negative)\n        - *r* = true positive / (true positive + false positive)\n\n        The f-measure is:\n\n        - *1/(alpha/p + (1-alpha)/r)*\n\n        With ``alpha = 0.5``, this reduces to:\n\n        - *2pr / (p + r)*\n\n        :param value: value used in the ConfusionMatrix\n        :param alpha: Ratio of the cost of false negative compared to false\n            positives. Defaults to 0.5, where the costs are equal.\n        :type alpha: float\n        :return: the F-measure corresponding to ``value``.\n        :rtype: float\n        '
        p = self.precision(value)
        r = self.recall(value)
        if p == 0.0 or r == 0.0:
            return 0.0
        return 1.0 / (alpha / p + (1 - alpha) / r)

    def evaluate(self, alpha=0.5, truncate=None, sort_by_count=False):
        if False:
            return 10
        '\n        Tabulate the **recall**, **precision** and **f-measure**\n        for each value in this confusion matrix.\n\n        >>> reference = "DET NN VB DET JJ NN NN IN DET NN".split()\n        >>> test = "DET VB VB DET NN NN NN IN DET NN".split()\n        >>> cm = ConfusionMatrix(reference, test)\n        >>> print(cm.evaluate())\n        Tag | Prec.  | Recall | F-measure\n        ----+--------+--------+-----------\n        DET | 1.0000 | 1.0000 | 1.0000\n         IN | 1.0000 | 1.0000 | 1.0000\n         JJ | 0.0000 | 0.0000 | 0.0000\n         NN | 0.7500 | 0.7500 | 0.7500\n         VB | 0.5000 | 1.0000 | 0.6667\n        <BLANKLINE>\n\n        :param alpha: Ratio of the cost of false negative compared to false\n            positives, as used in the f-measure computation. Defaults to 0.5,\n            where the costs are equal.\n        :type alpha: float\n        :param truncate: If specified, then only show the specified\n            number of values. Any sorting (e.g., sort_by_count)\n            will be performed before truncation. Defaults to None\n        :type truncate: int, optional\n        :param sort_by_count: Whether to sort the outputs on frequency\n            in the reference label. Defaults to False.\n        :type sort_by_count: bool, optional\n        :return: A tabulated recall, precision and f-measure string\n        :rtype: str\n        '
        tags = self._values
        if sort_by_count:
            tags = sorted(tags, key=lambda v: -sum(self._confusion[self._indices[v]]))
        if truncate:
            tags = tags[:truncate]
        tag_column_len = max(max((len(tag) for tag in tags)), 3)
        s = f"{' ' * (tag_column_len - 3)}Tag | Prec.  | Recall | F-measure\n{'-' * tag_column_len}-+--------+--------+-----------\n"
        for tag in tags:
            s += f'{tag:>{tag_column_len}} | {self.precision(tag):<6.4f} | {self.recall(tag):<6.4f} | {self.f_measure(tag, alpha=alpha):.4f}\n'
        return s

def demo():
    if False:
        while True:
            i = 10
    reference = 'DET NN VB DET JJ NN NN IN DET NN'.split()
    test = 'DET VB VB DET NN NN NN IN DET NN'.split()
    print('Reference =', reference)
    print('Test    =', test)
    print('Confusion matrix:')
    print(ConfusionMatrix(reference, test))
    print(ConfusionMatrix(reference, test).pretty_format(sort_by_count=True))
    print(ConfusionMatrix(reference, test).recall('VB'))
if __name__ == '__main__':
    demo()