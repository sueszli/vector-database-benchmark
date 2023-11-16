"""
A theorem prover that makes use of the external 'Prover9' package.
"""
import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import AllExpression, AndExpression, EqualityExpression, ExistsExpression, Expression, IffExpression, ImpExpression, NegatedExpression, OrExpression
p9_return_codes = {0: True, 1: '(FATAL)', 2: False, 3: '(MAX_MEGS)', 4: '(MAX_SECONDS)', 5: '(MAX_GIVEN)', 6: '(MAX_KEPT)', 7: '(ACTION)', 101: '(SIGSEGV)'}

class Prover9CommandParent:
    """
    A common base class used by both ``Prover9Command`` and ``MaceCommand``,
    which is responsible for maintaining a goal and a set of assumptions,
    and generating prover9-style input files from them.
    """

    def print_assumptions(self, output_format='nltk'):
        if False:
            print('Hello World!')
        '\n        Print the list of the current assumptions.\n        '
        if output_format.lower() == 'nltk':
            for a in self.assumptions():
                print(a)
        elif output_format.lower() == 'prover9':
            for a in convert_to_prover9(self.assumptions()):
                print(a)
        else:
            raise NameError("Unrecognized value for 'output_format': %s" % output_format)

class Prover9Command(Prover9CommandParent, BaseProverCommand):
    """
    A ``ProverCommand`` specific to the ``Prover9`` prover.  It contains
    the a print_assumptions() method that is used to print the list
    of assumptions in multiple formats.
    """

    def __init__(self, goal=None, assumptions=None, timeout=60, prover=None):
        if False:
            return 10
        '\n        :param goal: Input expression to prove\n        :type goal: sem.Expression\n        :param assumptions: Input expressions to use as assumptions in\n            the proof.\n        :type assumptions: list(sem.Expression)\n        :param timeout: number of seconds before timeout; set to 0 for\n            no timeout.\n        :type timeout: int\n        :param prover: a prover.  If not set, one will be created.\n        :type prover: Prover9\n        '
        if not assumptions:
            assumptions = []
        if prover is not None:
            assert isinstance(prover, Prover9)
        else:
            prover = Prover9(timeout)
        BaseProverCommand.__init__(self, prover, goal, assumptions)

    def decorate_proof(self, proof_string, simplify=True):
        if False:
            i = 10
            return i + 15
        '\n        :see BaseProverCommand.decorate_proof()\n        '
        if simplify:
            return self._prover._call_prooftrans(proof_string, ['striplabels'])[0].rstrip()
        else:
            return proof_string.rstrip()

class Prover9Parent:
    """
    A common class extended by both ``Prover9`` and ``Mace <mace.Mace>``.
    It contains the functionality required to convert NLTK-style
    expressions into Prover9-style expressions.
    """
    _binary_location = None

    def config_prover9(self, binary_location, verbose=False):
        if False:
            i = 10
            return i + 15
        if binary_location is None:
            self._binary_location = None
            self._prover9_bin = None
        else:
            name = 'prover9'
            self._prover9_bin = nltk.internals.find_binary(name, path_to_bin=binary_location, env_vars=['PROVER9'], url='https://www.cs.unm.edu/~mccune/prover9/', binary_names=[name, name + '.exe'], verbose=verbose)
            self._binary_location = self._prover9_bin.rsplit(os.path.sep, 1)

    def prover9_input(self, goal, assumptions):
        if False:
            print('Hello World!')
        '\n        :return: The input string that should be provided to the\n            prover9 binary.  This string is formed based on the goal,\n            assumptions, and timeout value of this object.\n        '
        s = ''
        if assumptions:
            s += 'formulas(assumptions).\n'
            for p9_assumption in convert_to_prover9(assumptions):
                s += '    %s.\n' % p9_assumption
            s += 'end_of_list.\n\n'
        if goal:
            s += 'formulas(goals).\n'
            s += '    %s.\n' % convert_to_prover9(goal)
            s += 'end_of_list.\n\n'
        return s

    def binary_locations(self):
        if False:
            i = 10
            return i + 15
        '\n        A list of directories that should be searched for the prover9\n        executables.  This list is used by ``config_prover9`` when searching\n        for the prover9 executables.\n        '
        return ['/usr/local/bin/prover9', '/usr/local/bin/prover9/bin', '/usr/local/bin', '/usr/bin', '/usr/local/prover9', '/usr/local/share/prover9']

    def _find_binary(self, name, verbose=False):
        if False:
            print('Hello World!')
        binary_locations = self.binary_locations()
        if self._binary_location is not None:
            binary_locations += [self._binary_location]
        return nltk.internals.find_binary(name, searchpath=binary_locations, env_vars=['PROVER9'], url='https://www.cs.unm.edu/~mccune/prover9/', binary_names=[name, name + '.exe'], verbose=verbose)

    def _call(self, input_str, binary, args=[], verbose=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call the binary with the given input.\n\n        :param input_str: A string whose contents are used as stdin.\n        :param binary: The location of the binary to call\n        :param args: A list of command-line arguments.\n        :return: A tuple (stdout, returncode)\n        :see: ``config_prover9``\n        '
        if verbose:
            print('Calling:', binary)
            print('Args:', args)
            print('Input:\n', input_str, '\n')
        cmd = [binary] + args
        try:
            input_str = input_str.encode('utf8')
        except AttributeError:
            pass
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        (stdout, stderr) = p.communicate(input=input_str)
        if verbose:
            print('Return code:', p.returncode)
            if stdout:
                print('stdout:\n', stdout, '\n')
            if stderr:
                print('stderr:\n', stderr, '\n')
        return (stdout.decode('utf-8'), p.returncode)

def convert_to_prover9(input):
    if False:
        return 10
    '\n    Convert a ``logic.Expression`` to Prover9 format.\n    '
    if isinstance(input, list):
        result = []
        for s in input:
            try:
                result.append(_convert_to_prover9(s.simplify()))
            except:
                print('input %s cannot be converted to Prover9 input syntax' % input)
                raise
        return result
    else:
        try:
            return _convert_to_prover9(input.simplify())
        except:
            print('input %s cannot be converted to Prover9 input syntax' % input)
            raise

def _convert_to_prover9(expression):
    if False:
        return 10
    '\n    Convert ``logic.Expression`` to Prover9 formatted string.\n    '
    if isinstance(expression, ExistsExpression):
        return 'exists ' + str(expression.variable) + ' ' + _convert_to_prover9(expression.term)
    elif isinstance(expression, AllExpression):
        return 'all ' + str(expression.variable) + ' ' + _convert_to_prover9(expression.term)
    elif isinstance(expression, NegatedExpression):
        return '-(' + _convert_to_prover9(expression.term) + ')'
    elif isinstance(expression, AndExpression):
        return '(' + _convert_to_prover9(expression.first) + ' & ' + _convert_to_prover9(expression.second) + ')'
    elif isinstance(expression, OrExpression):
        return '(' + _convert_to_prover9(expression.first) + ' | ' + _convert_to_prover9(expression.second) + ')'
    elif isinstance(expression, ImpExpression):
        return '(' + _convert_to_prover9(expression.first) + ' -> ' + _convert_to_prover9(expression.second) + ')'
    elif isinstance(expression, IffExpression):
        return '(' + _convert_to_prover9(expression.first) + ' <-> ' + _convert_to_prover9(expression.second) + ')'
    elif isinstance(expression, EqualityExpression):
        return '(' + _convert_to_prover9(expression.first) + ' = ' + _convert_to_prover9(expression.second) + ')'
    else:
        return str(expression)

class Prover9(Prover9Parent, Prover):
    _prover9_bin = None
    _prooftrans_bin = None

    def __init__(self, timeout=60):
        if False:
            print('Hello World!')
        self._timeout = timeout
        'The timeout value for prover9.  If a proof can not be found\n           in this amount of time, then prover9 will return false.\n           (Use 0 for no timeout.)'

    def _prove(self, goal=None, assumptions=None, verbose=False):
        if False:
            print('Hello World!')
        '\n        Use Prover9 to prove a theorem.\n        :return: A pair whose first element is a boolean indicating if the\n        proof was successful (i.e. returns value of 0) and whose second element\n        is the output of the prover.\n        '
        if not assumptions:
            assumptions = []
        (stdout, returncode) = self._call_prover9(self.prover9_input(goal, assumptions), verbose=verbose)
        return (returncode == 0, stdout)

    def prover9_input(self, goal, assumptions):
        if False:
            for i in range(10):
                print('nop')
        '\n        :see: Prover9Parent.prover9_input\n        '
        s = 'clear(auto_denials).\n'
        return s + Prover9Parent.prover9_input(self, goal, assumptions)

    def _call_prover9(self, input_str, args=[], verbose=False):
        if False:
            while True:
                i = 10
        '\n        Call the ``prover9`` binary with the given input.\n\n        :param input_str: A string whose contents are used as stdin.\n        :param args: A list of command-line arguments.\n        :return: A tuple (stdout, returncode)\n        :see: ``config_prover9``\n        '
        if self._prover9_bin is None:
            self._prover9_bin = self._find_binary('prover9', verbose)
        updated_input_str = ''
        if self._timeout > 0:
            updated_input_str += 'assign(max_seconds, %d).\n\n' % self._timeout
        updated_input_str += input_str
        (stdout, returncode) = self._call(updated_input_str, self._prover9_bin, args, verbose)
        if returncode not in [0, 2]:
            errormsgprefix = '%%ERROR:'
            if errormsgprefix in stdout:
                msgstart = stdout.index(errormsgprefix)
                errormsg = stdout[msgstart:].strip()
            else:
                errormsg = None
            if returncode in [3, 4, 5, 6]:
                raise Prover9LimitExceededException(returncode, errormsg)
            else:
                raise Prover9FatalException(returncode, errormsg)
        return (stdout, returncode)

    def _call_prooftrans(self, input_str, args=[], verbose=False):
        if False:
            i = 10
            return i + 15
        '\n        Call the ``prooftrans`` binary with the given input.\n\n        :param input_str: A string whose contents are used as stdin.\n        :param args: A list of command-line arguments.\n        :return: A tuple (stdout, returncode)\n        :see: ``config_prover9``\n        '
        if self._prooftrans_bin is None:
            self._prooftrans_bin = self._find_binary('prooftrans', verbose)
        return self._call(input_str, self._prooftrans_bin, args, verbose)

class Prover9Exception(Exception):

    def __init__(self, returncode, message):
        if False:
            print('Hello World!')
        msg = p9_return_codes[returncode]
        if message:
            msg += '\n%s' % message
        Exception.__init__(self, msg)

class Prover9FatalException(Prover9Exception):
    pass

class Prover9LimitExceededException(Prover9Exception):
    pass

def test_config():
    if False:
        while True:
            i = 10
    a = Expression.fromstring('(walk(j) & sing(j))')
    g = Expression.fromstring('walk(j)')
    p = Prover9Command(g, assumptions=[a])
    p._executable_path = None
    p.prover9_search = []
    p.prove()
    print(p.prove())
    print(p.proof())

def test_convert_to_prover9(expr):
    if False:
        print('Hello World!')
    '\n    Test that parsing works OK.\n    '
    for t in expr:
        e = Expression.fromstring(t)
        print(convert_to_prover9(e))

def test_prove(arguments):
    if False:
        while True:
            i = 10
    '\n    Try some proofs and exhibit the results.\n    '
    for (goal, assumptions) in arguments:
        g = Expression.fromstring(goal)
        alist = [Expression.fromstring(a) for a in assumptions]
        p = Prover9Command(g, assumptions=alist).prove()
        for a in alist:
            print('   %s' % a)
        print(f'|- {g}: {p}\n')
arguments = [('(man(x) <-> (not (not man(x))))', []), ('(not (man(x) & (not man(x))))', []), ('(man(x) | (not man(x)))', []), ('(man(x) & (not man(x)))', []), ('(man(x) -> man(x))', []), ('(not (man(x) & (not man(x))))', []), ('(man(x) | (not man(x)))', []), ('(man(x) -> man(x))', []), ('(man(x) <-> man(x))', []), ('(not (man(x) <-> (not man(x))))', []), ('mortal(Socrates)', ['all x.(man(x) -> mortal(x))', 'man(Socrates)']), ('((all x.(man(x) -> walks(x)) & man(Socrates)) -> some y.walks(y))', []), ('(all x.man(x) -> all x.man(x))', []), ('some x.all y.sees(x,y)', []), ('some e3.(walk(e3) & subj(e3, mary))', ['some e1.(see(e1) & subj(e1, john) & some e2.(pred(e1, e2) & walk(e2) & subj(e2, mary)))']), ('some x e1.(see(e1) & subj(e1, x) & some e2.(pred(e1, e2) & walk(e2) & subj(e2, mary)))', ['some e1.(see(e1) & subj(e1, john) & some e2.(pred(e1, e2) & walk(e2) & subj(e2, mary)))'])]
expressions = ['some x y.sees(x,y)', 'some x.(man(x) & walks(x))', '\\x.(man(x) & walks(x))', '\\x y.sees(x,y)', 'walks(john)', '\\x.big(x, \\y.mouse(y))', '(walks(x) & (runs(x) & (threes(x) & fours(x))))', '(walks(x) -> runs(x))', 'some x.(PRO(x) & sees(John, x))', 'some x.(man(x) & (not walks(x)))', 'all x.(man(x) -> walks(x))']

def spacer(num=45):
    if False:
        for i in range(10):
            print('nop')
    print('-' * num)

def demo():
    if False:
        return 10
    print('Testing configuration')
    spacer()
    test_config()
    print()
    print('Testing conversion to Prover9 format')
    spacer()
    test_convert_to_prover9(expressions)
    print()
    print('Testing proofs')
    spacer()
    test_prove(arguments)
if __name__ == '__main__':
    demo()