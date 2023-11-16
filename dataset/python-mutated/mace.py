"""
A model builder that makes use of the external 'Mace4' package.
"""
import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar

class MaceCommand(Prover9CommandParent, BaseModelBuilderCommand):
    """
    A ``MaceCommand`` specific to the ``Mace`` model builder.  It contains
    a print_assumptions() method that is used to print the list
    of assumptions in multiple formats.
    """
    _interpformat_bin = None

    def __init__(self, goal=None, assumptions=None, max_models=500, model_builder=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param goal: Input expression to prove\n        :type goal: sem.Expression\n        :param assumptions: Input expressions to use as assumptions in\n            the proof.\n        :type assumptions: list(sem.Expression)\n        :param max_models: The maximum number of models that Mace will try before\n            simply returning false. (Use 0 for no maximum.)\n        :type max_models: int\n        '
        if model_builder is not None:
            assert isinstance(model_builder, Mace)
        else:
            model_builder = Mace(max_models)
        BaseModelBuilderCommand.__init__(self, model_builder, goal, assumptions)

    @property
    def valuation(mbc):
        if False:
            while True:
                i = 10
        return mbc.model('valuation')

    def _convert2val(self, valuation_str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform the output file into an NLTK-style Valuation.\n\n        :return: A model if one is generated; None otherwise.\n        :rtype: sem.Valuation\n        '
        valuation_standard_format = self._transform_output(valuation_str, 'standard')
        val = []
        for line in valuation_standard_format.splitlines(False):
            l = line.strip()
            if l.startswith('interpretation'):
                num_entities = int(l[l.index('(') + 1:l.index(',')].strip())
            elif l.startswith('function') and l.find('_') == -1:
                name = l[l.index('(') + 1:l.index(',')].strip()
                if is_indvar(name):
                    name = name.upper()
                value = int(l[l.index('[') + 1:l.index(']')].strip())
                val.append((name, MaceCommand._make_model_var(value)))
            elif l.startswith('relation'):
                l = l[l.index('(') + 1:]
                if '(' in l:
                    name = l[:l.index('(')].strip()
                    values = [int(v.strip()) for v in l[l.index('[') + 1:l.index(']')].split(',')]
                    val.append((name, MaceCommand._make_relation_set(num_entities, values)))
                else:
                    name = l[:l.index(',')].strip()
                    value = int(l[l.index('[') + 1:l.index(']')].strip())
                    val.append((name, value == 1))
        return Valuation(val)

    @staticmethod
    def _make_relation_set(num_entities, values):
        if False:
            print('Hello World!')
        "\n        Convert a Mace4-style relation table into a dictionary.\n\n        :param num_entities: the number of entities in the model; determines the row length in the table.\n        :type num_entities: int\n        :param values: a list of 1's and 0's that represent whether a relation holds in a Mace4 model.\n        :type values: list of int\n        "
        r = set()
        for position in [pos for (pos, v) in enumerate(values) if v == 1]:
            r.add(tuple(MaceCommand._make_relation_tuple(position, values, num_entities)))
        return r

    @staticmethod
    def _make_relation_tuple(position, values, num_entities):
        if False:
            i = 10
            return i + 15
        if len(values) == 1:
            return []
        else:
            sublist_size = len(values) // num_entities
            sublist_start = position // sublist_size
            sublist_position = int(position % sublist_size)
            sublist = values[sublist_start * sublist_size:(sublist_start + 1) * sublist_size]
            return [MaceCommand._make_model_var(sublist_start)] + MaceCommand._make_relation_tuple(sublist_position, sublist, num_entities)

    @staticmethod
    def _make_model_var(value):
        if False:
            return 10
        '\n        Pick an alphabetic character as identifier for an entity in the model.\n\n        :param value: where to index into the list of characters\n        :type value: int\n        '
        letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'][value]
        num = value // 26
        return letter + str(num) if num > 0 else letter

    def _decorate_model(self, valuation_str, format):
        if False:
            print('Hello World!')
        "\n        Print out a Mace4 model using any Mace4 ``interpformat`` format.\n        See https://www.cs.unm.edu/~mccune/mace4/manual/ for details.\n\n        :param valuation_str: str with the model builder's output\n        :param format: str indicating the format for displaying\n        models. Defaults to 'standard' format.\n        :return: str\n        "
        if not format:
            return valuation_str
        elif format == 'valuation':
            return self._convert2val(valuation_str)
        else:
            return self._transform_output(valuation_str, format)

    def _transform_output(self, valuation_str, format):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform the output file into any Mace4 ``interpformat`` format.\n\n        :param format: Output format for displaying models.\n        :type format: str\n        '
        if format in ['standard', 'standard2', 'portable', 'tabular', 'raw', 'cooked', 'xml', 'tex']:
            return self._call_interpformat(valuation_str, [format])[0]
        else:
            raise LookupError('The specified format does not exist')

    def _call_interpformat(self, input_str, args=[], verbose=False):
        if False:
            return 10
        '\n        Call the ``interpformat`` binary with the given input.\n\n        :param input_str: A string whose contents are used as stdin.\n        :param args: A list of command-line arguments.\n        :return: A tuple (stdout, returncode)\n        :see: ``config_prover9``\n        '
        if self._interpformat_bin is None:
            self._interpformat_bin = self._modelbuilder._find_binary('interpformat', verbose)
        return self._modelbuilder._call(input_str, self._interpformat_bin, args, verbose)

class Mace(Prover9Parent, ModelBuilder):
    _mace4_bin = None

    def __init__(self, end_size=500):
        if False:
            while True:
                i = 10
        self._end_size = end_size
        'The maximum model size that Mace will try before\n           simply returning false. (Use -1 for no maximum.)'

    def _build_model(self, goal=None, assumptions=None, verbose=False):
        if False:
            print('Hello World!')
        '\n        Use Mace4 to build a first order model.\n\n        :return: ``True`` if a model was found (i.e. Mace returns value of 0),\n        else ``False``\n        '
        if not assumptions:
            assumptions = []
        (stdout, returncode) = self._call_mace4(self.prover9_input(goal, assumptions), verbose=verbose)
        return (returncode == 0, stdout)

    def _call_mace4(self, input_str, args=[], verbose=False):
        if False:
            return 10
        '\n        Call the ``mace4`` binary with the given input.\n\n        :param input_str: A string whose contents are used as stdin.\n        :param args: A list of command-line arguments.\n        :return: A tuple (stdout, returncode)\n        :see: ``config_prover9``\n        '
        if self._mace4_bin is None:
            self._mace4_bin = self._find_binary('mace4', verbose)
        updated_input_str = ''
        if self._end_size > 0:
            updated_input_str += 'assign(end_size, %d).\n\n' % self._end_size
        updated_input_str += input_str
        return self._call(updated_input_str, self._mace4_bin, args, verbose)

def spacer(num=30):
    if False:
        while True:
            i = 10
    print('-' * num)

def decode_result(found):
    if False:
        print('Hello World!')
    '\n    Decode the result of model_found()\n\n    :param found: The output of model_found()\n    :type found: bool\n    '
    return {True: 'Countermodel found', False: 'No countermodel found', None: 'None'}[found]

def test_model_found(arguments):
    if False:
        return 10
    '\n    Try some proofs and exhibit the results.\n    '
    for (goal, assumptions) in arguments:
        g = Expression.fromstring(goal)
        alist = [lp.parse(a) for a in assumptions]
        m = MaceCommand(g, assumptions=alist, max_models=50)
        found = m.build_model()
        for a in alist:
            print('   %s' % a)
        print(f'|- {g}: {decode_result(found)}\n')

def test_build_model(arguments):
    if False:
        i = 10
        return i + 15
    '\n    Try to build a ``nltk.sem.Valuation``.\n    '
    g = Expression.fromstring('all x.man(x)')
    alist = [Expression.fromstring(a) for a in ['man(John)', 'man(Socrates)', 'man(Bill)', 'some x.(-(x = John) & man(x) & sees(John,x))', 'some x.(-(x = Bill) & man(x))', 'all x.some y.(man(x) -> gives(Socrates,x,y))']]
    m = MaceCommand(g, assumptions=alist)
    m.build_model()
    spacer()
    print('Assumptions and Goal')
    spacer()
    for a in alist:
        print('   %s' % a)
    print(f'|- {g}: {decode_result(m.build_model())}\n')
    spacer()
    print('Valuation')
    spacer()
    print(m.valuation, '\n')

def test_transform_output(argument_pair):
    if False:
        print('Hello World!')
    '\n    Transform the model into various Mace4 ``interpformat`` formats.\n    '
    g = Expression.fromstring(argument_pair[0])
    alist = [lp.parse(a) for a in argument_pair[1]]
    m = MaceCommand(g, assumptions=alist)
    m.build_model()
    for a in alist:
        print('   %s' % a)
    print(f'|- {g}: {m.build_model()}\n')
    for format in ['standard', 'portable', 'xml', 'cooked']:
        spacer()
        print("Using '%s' format" % format)
        spacer()
        print(m.model(format=format))

def test_make_relation_set():
    if False:
        return 10
    print(MaceCommand._make_relation_set(num_entities=3, values=[1, 0, 1]) == {('c',), ('a',)})
    print(MaceCommand._make_relation_set(num_entities=3, values=[0, 0, 0, 0, 0, 0, 1, 0, 0]) == {('c', 'a')})
    print(MaceCommand._make_relation_set(num_entities=2, values=[0, 0, 1, 0, 0, 0, 1, 0]) == {('a', 'b', 'a'), ('b', 'b', 'a')})
arguments = [('mortal(Socrates)', ['all x.(man(x) -> mortal(x))', 'man(Socrates)']), ('(not mortal(Socrates))', ['all x.(man(x) -> mortal(x))', 'man(Socrates)'])]

def demo():
    if False:
        i = 10
        return i + 15
    test_model_found(arguments)
    test_build_model(arguments)
    test_transform_output(arguments[1])
if __name__ == '__main__':
    demo()