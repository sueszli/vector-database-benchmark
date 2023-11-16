from collections import defaultdict

class MinimalSet:
    """
    Find contexts where more than one possible target value can
    appear.  E.g. if targets are word-initial letters, and contexts
    are the remainders of words, then we would like to find cases like
    "fat" vs "cat", and "training" vs "draining".  If targets are
    parts-of-speech and contexts are words, then we would like to find
    cases like wind (noun) 'air in rapid motion', vs wind (verb)
    'coil, wrap'.
    """

    def __init__(self, parameters=None):
        if False:
            print('Hello World!')
        '\n        Create a new minimal set.\n\n        :param parameters: The (context, target, display) tuples for the item\n        :type parameters: list(tuple(str, str, str))\n        '
        self._targets = set()
        self._contexts = set()
        self._seen = defaultdict(set)
        self._displays = {}
        if parameters:
            for (context, target, display) in parameters:
                self.add(context, target, display)

    def add(self, context, target, display):
        if False:
            while True:
                i = 10
        '\n        Add a new item to the minimal set, having the specified\n        context, target, and display form.\n\n        :param context: The context in which the item of interest appears\n        :type context: str\n        :param target: The item of interest\n        :type target: str\n        :param display: The information to be reported for each item\n        :type display: str\n        '
        self._seen[context].add(target)
        self._contexts.add(context)
        self._targets.add(target)
        self._displays[context, target] = display

    def contexts(self, minimum=2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine which contexts occurred with enough distinct targets.\n\n        :param minimum: the minimum number of distinct target forms\n        :type minimum: int\n        :rtype: list\n        '
        return [c for c in self._contexts if len(self._seen[c]) >= minimum]

    def display(self, context, target, default=''):
        if False:
            while True:
                i = 10
        if (context, target) in self._displays:
            return self._displays[context, target]
        else:
            return default

    def display_all(self, context):
        if False:
            print('Hello World!')
        result = []
        for target in self._targets:
            x = self.display(context, target)
            if x:
                result.append(x)
        return result

    def targets(self):
        if False:
            i = 10
            return i + 15
        return self._targets