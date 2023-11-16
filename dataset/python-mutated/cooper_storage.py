from nltk.parse import load_parser
from nltk.parse.featurechart import InstantiateVarsChart
from nltk.sem.logic import ApplicationExpression, LambdaExpression, Variable

class CooperStore:
    """
    A container for handling quantifier ambiguity via Cooper storage.
    """

    def __init__(self, featstruct):
        if False:
            print('Hello World!')
        '\n        :param featstruct: The value of the ``sem`` node in a tree from\n            ``parse_with_bindops()``\n        :type featstruct: FeatStruct (with features ``core`` and ``store``)\n\n        '
        self.featstruct = featstruct
        self.readings = []
        try:
            self.core = featstruct['CORE']
            self.store = featstruct['STORE']
        except KeyError:
            print('%s is not a Cooper storage structure' % featstruct)

    def _permute(self, lst):
        if False:
            i = 10
            return i + 15
        '\n        :return: An iterator over the permutations of the input list\n        :type lst: list\n        :rtype: iter\n        '
        remove = lambda lst0, index: lst0[:index] + lst0[index + 1:]
        if lst:
            for (index, x) in enumerate(lst):
                for y in self._permute(remove(lst, index)):
                    yield ((x,) + y)
        else:
            yield ()

    def s_retrieve(self, trace=False):
        if False:
            while True:
                i = 10
        '\n        Carry out S-Retrieval of binding operators in store. If hack=True,\n        serialize the bindop and core as strings and reparse. Ugh.\n\n        Each permutation of the store (i.e. list of binding operators) is\n        taken to be a possible scoping of quantifiers. We iterate through the\n        binding operators in each permutation, and successively apply them to\n        the current term, starting with the core semantic representation,\n        working from the inside out.\n\n        Binding operators are of the form::\n\n             bo(\\P.all x.(man(x) -> P(x)),z1)\n        '
        for (perm, store_perm) in enumerate(self._permute(self.store)):
            if trace:
                print('Permutation %s' % (perm + 1))
            term = self.core
            for bindop in store_perm:
                (quant, varex) = tuple(bindop.args)
                term = ApplicationExpression(quant, LambdaExpression(varex.variable, term))
                if trace:
                    print('  ', term)
                term = term.simplify()
            self.readings.append(term)

def parse_with_bindops(sentence, grammar=None, trace=0):
    if False:
        return 10
    '\n    Use a grammar with Binding Operators to parse a sentence.\n    '
    if not grammar:
        grammar = 'grammars/book_grammars/storage.fcfg'
    parser = load_parser(grammar, trace=trace, chart_class=InstantiateVarsChart)
    tokens = sentence.split()
    return list(parser.parse(tokens))

def demo():
    if False:
        i = 10
        return i + 15
    from nltk.sem import cooper_storage as cs
    sentence = 'every girl chases a dog'
    print()
    print("Analysis of sentence '%s'" % sentence)
    print('=' * 50)
    trees = cs.parse_with_bindops(sentence, trace=0)
    for tree in trees:
        semrep = cs.CooperStore(tree.label()['SEM'])
        print()
        print('Binding operators:')
        print('-' * 15)
        for s in semrep.store:
            print(s)
        print()
        print('Core:')
        print('-' * 15)
        print(semrep.core)
        print()
        print('S-Retrieval:')
        print('-' * 15)
        semrep.s_retrieve(trace=True)
        print('Readings:')
        print('-' * 15)
        for (i, reading) in enumerate(semrep.readings):
            print(f'{i + 1}: {reading}')
if __name__ == '__main__':
    demo()