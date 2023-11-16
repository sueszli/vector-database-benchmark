import collections
import fcntl
import os
import shlex
import shutil
import time
from abc import abstractmethod
from random import shuffle
from subprocess import PIPE, Popen, check_output
from typing import Any, Sequence, List
from . import operators as Operators
from .constraints import *
from .visitors import *
from ...exceptions import SolverError, SolverUnknown, TooManySolutions, SmtlibError
from ...utils import config
logger = logging.getLogger(__name__)
consts = config.get_group('smt')
consts.add('timeout', default=120, description='Timeout, in seconds, for each Z3 invocation')
consts.add('memory', default=1024 * 8, description='Max memory for Z3 to use (in Megabytes)')
consts.add('maxsolutions', default=10000, description='Maximum solutions to provide when solving for all values')
consts.add('z3_bin', default='z3', description='Z3 solver binary to use')
consts.add('cvc4_bin', default='cvc4', description='CVC4 solver binary to use')
consts.add('yices_bin', default='yices-smt2', description='Yices solver binary to use')
consts.add('boolector_bin', default='boolector', description='Boolector solver binary to use')
consts.add('defaultunsat', default=True, description='Consider solver timeouts as unsat core')
consts.add('optimize', default=True, description='Use smtlib command optimize to find min/max if available')
RE_GET_EXPR_VALUE_ALL = re.compile('\\(([a-zA-Z0-9_]*)[ \\n\\s]*(#b[0-1]*|#x[0-9a-fA-F]*|[(]?_ bv[0-9]* [0-9]*|true|false)\\)')
RE_GET_EXPR_VALUE_FMT_BIN = re.compile('\\(\\((?P<expr>(.*))[ \\n\\s]*#b(?P<value>([0-1]*))\\)\\)')
RE_GET_EXPR_VALUE_FMT_DEC = re.compile('\\(\\((?P<expr>(.*))\\ \\(_\\ bv(?P<value>(\\d*))\\ \\d*\\)\\)\\)')
RE_GET_EXPR_VALUE_FMT_HEX = re.compile('\\(\\((?P<expr>(.*))\\ #x(?P<value>([0-9a-fA-F]*))\\)\\)')
RE_OBJECTIVES_EXPR_VALUE = re.compile('\\(objectives.*\\((?P<expr>.*) (?P<value>\\d*)\\).*\\).*', re.MULTILINE | re.DOTALL)
RE_MIN_MAX_OBJECTIVE_EXPR_VALUE = re.compile('(?P<expr>.*?)\\s+\\|->\\s+(?P<value>.*)', re.DOTALL)
SOLVER_STATS = {'unknown': 0, 'timeout': 0}

class SolverType(config.ConfigEnum):
    """Used as configuration constant for choosing solver flavor"""
    z3 = 'z3'
    cvc4 = 'cvc4'
    yices = 'yices'
    auto = 'auto'
    portfolio = 'portfolio'
    boolector = 'boolector'
consts.add('solver', default=SolverType.auto, description='Choose default smtlib2 solver (z3, yices, cvc4, boolector, portfolio, auto)')

def _convert(v):
    if False:
        while True:
            i = 10
    r = None
    if v == 'true':
        r = True
    elif v == 'false':
        r = False
    elif v.startswith('#b'):
        r = int(v[2:], 2)
    elif v.startswith('#x'):
        r = int(v[2:], 16)
    elif v.startswith('_ bv'):
        r = int(v[len('_ bv'):-len(' 256')], 10)
    elif v.startswith('(_ bv'):
        v = v[len('(_ bv'):]
        r = int(v[:v.find(' ')], 10)
    assert r is not None
    return r

class SingletonMixin(object):
    __singleton_instances: Dict[Tuple[int, int], 'SingletonMixin'] = {}

    @classmethod
    def instance(cls):
        if False:
            while True:
                i = 10
        tid = threading.get_ident()
        pid = os.getpid()
        if not (pid, tid) in cls.__singleton_instances:
            cls.__singleton_instances[pid, tid] = cls()
        return cls.__singleton_instances[pid, tid]

class SolverException(SmtlibError):
    """
    Solver exception
    """
    pass

class Solver(SingletonMixin):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def optimize(self, constraints, X, operation, M=10000):
        if False:
            while True:
                i = 10
        '\n        Iteratively finds the maximum or minimal value for the operation\n        (Normally Operators.UGT or Operators.ULT)\n\n        :param constraints: the constraints set\n        :param X: a symbol or expression\n        :param M: maximum number of iterations allowed\n        '
        raise SmtlibError('Abstract method not implemented')

    def check(self, constraints) -> bool:
        if False:
            print('Hello World!')
        'Check if given constraints can be valid'
        return self.can_be_true(constraints, True)

    def can_be_true(self, constraints, expression=True) -> bool:
        if False:
            print('Hello World!')
        'Check if given expression could be valid'
        raise SolverException('Abstract method not implemented')

    def must_be_true(self, constraints, expression) -> bool:
        if False:
            while True:
                i = 10
        'Check if expression is True and that it can not be False with current constraints'
        solutions = self.get_all_values(constraints, expression, maxcnt=2, silent=True)
        return solutions == [True]

    def get_all_values(self, constraints, x, maxcnt=10000, silent=False):
        if False:
            print('Hello World!')
        'Returns a list with all the possible values for the symbol x'
        raise SolverException('Abstract method not implemented')

    def get_value(self, constraints, expression):
        if False:
            for i in range(10):
                print('nop')
        'Ask the solver for one possible result of given expression using given set of constraints.'
        raise SolverException('Abstract method not implemented')

    def max(self, constraints, X: BitVec, M=10000):
        if False:
            print('Hello World!')
        '\n        Iteratively finds the maximum value for a symbol within given constraints.\n        :param X: a symbol or expression\n        :param M: maximum number of iterations allowed\n        '
        assert isinstance(X, BitVec)
        return self.optimize(constraints, X, 'maximize', M)

    def min(self, constraints, X: BitVec, M=10000):
        if False:
            return 10
        '\n        Iteratively finds the minimum value for a symbol within given constraints.\n\n        :param constraints: constraints that the expression must fulfil\n        :param X: a symbol or expression\n        :param M: maximum number of iterations allowed\n        '
        assert isinstance(X, BitVec)
        return self.optimize(constraints, X, 'minimize', M)

    def minmax(self, constraints, x, iters=10000):
        if False:
            i = 10
            return i + 15
        'Returns the min and max possible values for x within given constraints'
        if issymbolic(x):
            m = self.min(constraints, x, iters)
            M = self.max(constraints, x, iters)
            return (m, M)
        else:
            return (x, x)
Version = collections.namedtuple('Version', 'major minor patch')

class SmtlibProc:

    def __init__(self, command: str, debug: bool=False):
        if False:
            i = 10
            return i + 15
        'Single smtlib interactive process\n\n        :param command: the shell command to execute\n        :param debug: log all messaging\n        '
        self._proc: Optional[Popen] = None
        self._command = command
        self._debug = debug
        self._last_buf = ''

    def start(self):
        if False:
            return 10
        'Spawns POpen solver process'
        if self._proc is not None:
            return
        self._proc = Popen(shlex.split(self._command), stdin=PIPE, stdout=PIPE, universal_newlines=True, close_fds=True)
        fl = fcntl.fcntl(self._proc.stdout, fcntl.F_GETFL)
        fcntl.fcntl(self._proc.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        self._last_buf = ''

    def stop(self):
        if False:
            i = 10
            return i + 15
        "\n        Stops the solver process by:\n        - sending a SIGKILL signal,\n        - waiting till the process terminates (so we don't leave a zombie process)\n        "
        if self._proc is None:
            return
        if self._proc.returncode is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def send(self, cmd: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Send a string to the solver.\n\n        :param cmd: a SMTLIBv2 command (ex. (check-sat))\n        '
        if self._debug:
            logger.debug('>%s', cmd)
        assert self._proc is not None
        try:
            self._proc.stdout.flush()
            self._proc.stdin.write(f'{cmd}\n')
            self._proc.stdin.flush()
        except (BrokenPipeError, IOError) as e:
            logger.critical(f'Solver encountered an error trying to send commands: {e}.\n\tOutput: {self._proc.stdout}\n\n\tStderr: {self._proc.stderr}')
            raise e

    def recv(self, wait=True) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'Reads the response from the smtlib solver\n\n        :param wait: a boolean that indicate to wait with a blocking call\n        until the results are available. Otherwise, it returns None if the solver\n        does not respond.\n\n        '
        tries = 0
        timeout = 0.0
        buf = ''
        if self._last_buf != '':
            buf = buf + self._last_buf
        while True:
            try:
                buf = buf + self._proc.stdout.read()
                buf = buf.strip()
            except TypeError:
                if not wait:
                    if buf != '':
                        self._last_buf = buf
                    return None
                else:
                    tries += 1
            if buf == '':
                continue
            (lparen, rparen) = map(sum, zip(*((c == '(', c == ')') for c in buf)))
            if lparen == rparen and buf != '':
                break
            if tries > 3:
                time.sleep(timeout)
                timeout += 0.1
        buf = buf.strip()
        self._last_buf = ''
        if '(error' in buf:
            raise SolverException(f'Solver error: {buf}')
        if self._debug:
            logger.debug('<%s', buf)
        return buf

    def _restart(self) -> None:
        if False:
            while True:
                i = 10
        'Auxiliary to start or restart the external solver'
        self.stop()
        self.start()

    def is_started(self):
        if False:
            print('Hello World!')
        return self._proc is not None

    def clear_buffers(self):
        if False:
            i = 10
            return i + 15
        self._proc.stdout.flush()
        self._proc.stdin.flush()

class SMTLIBSolver(Solver):
    ncores: Optional[int] = None
    sname: Optional[str] = None

    @classmethod
    @abstractmethod
    def command(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def inits(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def __init__(self, command: str, init: Sequence[str]=None, support_reset: bool=False, support_minmax: bool=False, support_pushpop: bool=False, multiple_check: bool=True, debug: bool=False):
        if False:
            while True:
                i = 10
        '\n        Build a smtlib solver instance.\n        This is implemented using an external solver (via a subprocess).\n        '
        super().__init__()
        self._smtlib: SmtlibProc = SmtlibProc(command, debug)
        if init is None:
            init = tuple()
        self._init = init
        self._support_minmax = support_minmax
        self._support_reset = support_reset
        self._support_pushpop = support_pushpop
        self._multiple_check = multiple_check
        if not self._support_pushpop:
            setattr(self, '_push', None)
            setattr(self, '_pop', None)
        if self._support_minmax and consts.optimize:
            setattr(self, 'optimize', self._optimize_fancy)
        else:
            setattr(self, 'optimize', self._optimize_generic)
        self._smtlib.start()

    def _reset(self, constraints: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Auxiliary method to reset the smtlib external solver to initial defaults'
        if self._support_reset:
            self._smtlib.start()
            self._smtlib.send('(reset)')
        else:
            self._smtlib.stop()
            self._smtlib.start()
        self._smtlib.clear_buffers()
        for cfg in self._init:
            self._smtlib.send(cfg)
        if constraints is not None:
            self._smtlib.send(constraints)

    def _is_sat(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Check the satisfiability of the current state\n\n        :return: whether current state is satisfiable or not.\n        '
        start = time.time()
        self._smtlib.send('(check-sat)')
        status = self._smtlib.recv()
        assert status is not None
        logger.debug('Check took %s seconds (%s)', time.time() - start, status)
        if 'ALARM TRIGGERED' in status:
            return False
        if status not in ('sat', 'unsat', 'unknown'):
            raise SolverError(status)
        if consts.defaultunsat:
            if status == 'unknown':
                logger.info('Found an unknown core, probably a solver timeout')
                SOLVER_STATS['timeout'] += 1
                status = 'unsat'
                raise SolverUnknown(status)
        if status == 'unknown':
            SOLVER_STATS['unknown'] += 1
            raise SolverUnknown(status)
        else:
            assert self.sname is not None
            SOLVER_STATS.setdefault(self.sname, 0)
            SOLVER_STATS[self.sname] += 1
        return status == 'sat'

    def _assert(self, expression: Bool):
        if False:
            for i in range(10):
                print('nop')
        'Auxiliary method to send an assert'
        smtlib = translate_to_smtlib(expression)
        self._smtlib.send(f'(assert {smtlib})')

    def __getvalue_bv(self, expression_str: str) -> int:
        if False:
            return 10
        self._smtlib.send(f'(get-value ({expression_str}))')
        t = self._smtlib.recv()
        assert t is not None
        base = 2
        m = RE_GET_EXPR_VALUE_FMT_BIN.match(t)
        if m is None:
            m = RE_GET_EXPR_VALUE_FMT_DEC.match(t)
            base = 10
        if m is None:
            m = RE_GET_EXPR_VALUE_FMT_HEX.match(t)
            base = 16
        if m is None:
            raise SolverError(f"I don't know how to parse the value {str(t)} from {expression_str}")
        (expr, value) = (m.group('expr'), m.group('value'))
        return int(value, base)

    def __getvalue_bool(self, expression_str):
        if False:
            return 10
        self._smtlib.send(f'(get-value ({expression_str}))')
        ret = self._smtlib.recv()
        return {'true': True, 'false': False, '#b0': False, '#b1': True}[ret[2:-2].split(' ')[1]]

    def __getvalue_all(self, expressions_str: List[str], is_bv: List[bool]) -> Dict[str, int]:
        if False:
            while True:
                i = 10
        all_expressions_str = ' '.join(expressions_str)
        self._smtlib.send(f'(get-value ({all_expressions_str}))')
        ret_solver: Optional[str] = self._smtlib.recv()
        assert ret_solver is not None
        return_values = re.findall(RE_GET_EXPR_VALUE_ALL, ret_solver)
        return {value[0]: _convert(value[1]) for value in return_values}

    def _getvalue(self, expression) -> Union[int, bool, bytes]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Ask the solver for one possible assignment for given expression using current set of constraints.\n        The current set of expressions must be sat.\n\n        NOTE: This is an internal method: it uses the current solver state (set of constraints!).\n        '
        if not issymbolic(expression):
            return expression
        if isinstance(expression, Array):
            result = bytearray()
            for c in expression:
                expression_str = translate_to_smtlib(c)
                result.append(self.__getvalue_bv(expression_str))
            return bytes(result)
        elif isinstance(expression, BoolVariable):
            return self.__getvalue_bool(expression.name)
        elif isinstance(expression, BitVecVariable):
            return self.__getvalue_bv(expression.name)
        raise NotImplementedError(f'_getvalue only implemented for Bool, BitVec and Array. Got {type(expression)}')

    def _push(self):
        if False:
            return 10
        'Pushes and save the current constraint store and state.'
        self._smtlib.send('(push 1)')

    def _pop(self):
        if False:
            for i in range(10):
                print('nop')
        'Recall the last pushed constraint store and state.'
        self._smtlib.send('(pop 1)')

    @lru_cache(maxsize=32)
    def can_be_true(self, constraints: ConstraintSet, expression: Union[bool, Bool]=True) -> bool:
        if False:
            return 10
        'Check if two potentially symbolic values can be equal'
        if isinstance(expression, bool):
            if not expression:
                return expression
            else:
                self._reset(constraints.to_string())
                return self._is_sat()
        with constraints as temp_cs:
            temp_cs.add(expression)
            self._reset(temp_cs.to_string())
            return self._is_sat()

    def _optimize_generic(self, constraints: ConstraintSet, x: BitVec, goal: str, max_iter=10000):
        if False:
            print('Hello World!')
        "\n        Iteratively finds the maximum or minimum value for the operation\n        (Normally Operators.UGT or Operators.ULT)\n\n        :param constraints: constraints to take into account\n        :param x: a symbol or expression\n        :param goal: goal to achieve, either 'maximize' or 'minimize'\n        :param max_iter: maximum number of iterations allowed\n        "
        assert goal in ('maximize', 'minimize')
        operation = {'maximize': Operators.UGE, 'minimize': Operators.ULE}[goal]
        last_value: Optional[Union[int, bool, bytes]] = None
        start = time.time()
        with constraints as temp_cs:
            X = temp_cs.new_bitvec(x.size)
            temp_cs.add(X == x)
            self._reset(temp_cs.to_string())
            if not self._is_sat():
                raise SolverException('UNSAT')
            last_value = self._getvalue(X)
            self._assert(operation(X, last_value))
            if goal == 'maximize':
                (m, M) = (last_value, (1 << X.size) - 1)
            else:
                (m, M) = (0, last_value)
            L = None
            while L not in (M, m):
                L = (m + M) // 2
                self._assert(operation(X, L))
                sat = self._is_sat()
                if goal == 'maximize' and sat or (goal == 'minimize' and (not sat)):
                    m = L
                else:
                    M = L
                if time.time() - start > consts.timeout:
                    SOLVER_STATS['timeout'] += 1
                    raise SolverError('Timeout')
        with constraints as temp_cs:
            X = temp_cs.new_bitvec(x.size)
            temp_cs.add(X == x)
            self._reset(temp_cs.to_string())
            self._assert(Operators.UGE(X, m))
            self._assert(Operators.ULE(X, M))
            last_value = None
            i = 0
            while self._is_sat():
                last_value = self._getvalue(X)
                self._assert(operation(X, last_value))
                self._assert(X != last_value)
                i = i + 1
                if i > max_iter:
                    SOLVER_STATS['unknown'] += 1
                    raise SolverError('Optimizing error, maximum number of iterations was reached')
                if time.time() - start > consts.timeout:
                    SOLVER_STATS['timeout'] += 1
                    raise SolverError('Timeout')
            if last_value is not None:
                return last_value
            SOLVER_STATS['unknown'] += 1
            raise SolverError('Optimizing error, unsat or unknown core')

    @lru_cache(maxsize=32)
    def get_all_values(self, constraints: ConstraintSet, expression, maxcnt: Optional[int]=None, silent: bool=False):
        if False:
            return 10
        'Returns a list with all the possible values for the symbol x'
        if not isinstance(expression, Expression):
            return [expression]
        assert isinstance(expression, Expression)
        expression = simplify(expression)
        if maxcnt is None:
            maxcnt = consts.maxsolutions
            if isinstance(expression, Bool) and consts.maxsolutions > 1:
                maxcnt = 2
                silent = True
        with constraints as temp_cs:
            if isinstance(expression, Bool):
                var = temp_cs.new_bool()
            elif isinstance(expression, BitVec):
                var = temp_cs.new_bitvec(expression.size)
            elif isinstance(expression, Array):
                var = temp_cs.new_array(index_max=expression.index_max, value_bits=expression.value_bits, taint=expression.taint).array
            else:
                raise NotImplementedError(f'get_all_values only implemented for {type(expression)} expression type.')
            temp_cs.add(var == expression)
            self._reset(temp_cs.to_string())
            result = []
            start = time.time()
            while self._is_sat():
                value = self._getvalue(var)
                result.append(value)
                if len(result) >= maxcnt:
                    if silent:
                        break
                    else:
                        raise TooManySolutions(result)
                if time.time() - start > consts.timeout:
                    SOLVER_STATS['timeout'] += 1
                    if silent:
                        logger.info('Timeout searching for all solutions')
                        return list(result)
                    raise SolverError('Timeout')
                if self._multiple_check:
                    self._smtlib.send(f'(assert {translate_to_smtlib(var != value)})')
                else:
                    temp_cs.add(var != value)
                    self._reset(temp_cs.to_string())
            return list(result)

    def _optimize_fancy(self, constraints: ConstraintSet, x: BitVec, goal: str, max_iter=10000):
        if False:
            print('Hello World!')
        "\n        Iteratively finds the maximum or minimum value for the operation\n        (Normally Operators.UGT or Operators.ULT)\n\n        :param constraints: constraints to take into account\n        :param x: a symbol or expression\n        :param goal: goal to achieve, either 'maximize' or 'minimize'\n        :param max_iter: maximum number of iterations allowed\n        "
        assert goal in ('maximize', 'minimize')
        operation = {'maximize': Operators.UGE, 'minimize': Operators.ULE}[goal]
        with constraints as temp_cs:
            X = temp_cs.new_bitvec(x.size)
            temp_cs.add(X == x)
            aux = temp_cs.new_bitvec(X.size, name='optimized_')
            self._reset(temp_cs.to_string())
            self._assert(operation(X, aux))
            self._smtlib.send('(%s %s)' % (goal, aux.name))
            self._smtlib.send('(check-sat)')
            _status = self._smtlib.recv()
            assert self.sname is not None
            SOLVER_STATS.setdefault(self.sname, 0)
            SOLVER_STATS[self.sname] += 1
            if _status == 'sat':
                return self._getvalue(aux)
            raise SolverError('Optimize failed')

    def get_value(self, constraints: ConstraintSet, *expressions):
        if False:
            print('Hello World!')
        values = self.get_value_in_batch(constraints, expressions)
        if len(expressions) == 1:
            return values[0]
        else:
            return values

    def get_value_in_batch(self, constraints: ConstraintSet, expressions):
        if False:
            print('Hello World!')
        '\n        Ask the solver for one possible result of given expressions using\n        given set of constraints.\n        '
        values: List[Any] = [None] * len(expressions)
        start = time.time()
        with constraints.related_to(*expressions) as temp_cs:
            vars: List[Any] = []
            for (idx, expression) in enumerate(expressions):
                if not issymbolic(expression):
                    values[idx] = expression
                    vars.append(None)
                    continue
                assert isinstance(expression, (Bool, BitVec, Array))
                if isinstance(expression, Bool):
                    var = temp_cs.new_bool()
                    vars.append(var)
                    temp_cs.add(var == expression)
                elif isinstance(expression, BitVec):
                    var = temp_cs.new_bitvec(expression.size)
                    vars.append(var)
                    temp_cs.add(var == expression)
                elif isinstance(expression, Array):
                    var = []
                    for i in range(expression.index_max):
                        subvar = temp_cs.new_bitvec(expression.value_bits)
                        var.append(subvar)
                        temp_cs.add(subvar == simplify(expression[i]))
                    vars.append(var)
            self._reset(temp_cs.to_string())
            if not self._is_sat():
                raise SolverError('Solver could not find a value for expression under current constraint set')
            values_to_ask: List[str] = []
            is_bv: List[bool] = []
            for (idx, expression) in enumerate(expressions):
                if not issymbolic(expression):
                    continue
                var = vars[idx]
                if isinstance(expression, Bool):
                    values_to_ask.append(var.name)
                    is_bv.append(False)
                if isinstance(expression, BitVec):
                    values_to_ask.append(var.name)
                    is_bv.append(True)
                if isinstance(expression, Array):
                    for i in range(expression.index_max):
                        values_to_ask.append(var[i].name)
                        is_bv.append(True)
            if values_to_ask == []:
                return values
            values_returned = self.__getvalue_all(values_to_ask, is_bv)
            for (idx, expression) in enumerate(expressions):
                if not issymbolic(expression):
                    continue
                var = vars[idx]
                if isinstance(expression, Bool):
                    values[idx] = values_returned[var.name]
                if isinstance(expression, BitVec):
                    if var.name not in values_returned:
                        logger.error('var.name', var.name, 'not in values_returned', values_returned)
                    values[idx] = values_returned[var.name]
                if isinstance(expression, Array):
                    result = []
                    for i in range(expression.index_max):
                        result.append(values_returned[var[i].name])
                    values[idx] = bytes(result)
            if time.time() - start > consts.timeout:
                SOLVER_STATS['timeout'] += 1
                raise SolverError('Timeout')
        return values

class Z3Solver(SMTLIBSolver):
    sname = 'z3'

    @classmethod
    def command(self) -> str:
        if False:
            return 10
        return f'{consts.z3_bin} -t:{consts.timeout * 1000} -memory:{consts.memory} -smt2 -in'

    @classmethod
    def inits(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return ['(set-logic QF_AUFBV)', '(set-option :global-decls false)', '(set-option :tactic.solve_eqs.context_solve false)']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a Z3 solver instance.\n        This is implemented using an external z3 solver (via a subprocess).\n        See https://github.com/Z3Prover/z3\n        '
        command = self.command()
        self.ncores = 1
        (init, support_minmax, support_reset, multiple_check) = self.__autoconfig()
        super().__init__(command=command, init=init, support_minmax=support_minmax, support_reset=support_reset, multiple_check=multiple_check, support_pushpop=True, debug=False)

    def __autoconfig(self):
        if False:
            return 10
        init = self.inits()
        self.version = self._solver_version()
        support_reset = True
        support_minmax = self.version >= Version(4, 4, 1)
        multiple_check = self.version < Version(4, 8, 7)
        return (init, support_minmax, support_reset, multiple_check)

    def _solver_version(self) -> Version:
        if False:
            while True:
                i = 10
        "\n        If we fail to parse the version, we assume z3's output has changed, meaning it's a newer\n        version than what's used now, and therefore ok.\n\n        Anticipated version_cmd_output format: 'Z3 version 4.4.2'\n                                               'Z3 version 4.4.5 - 64 bit - build hashcode $Z3GITHASH'\n        "
        try:
            received_version = check_output([f'{consts.z3_bin}', '--version'])
            Z3VERSION = re.compile('.*(?P<major>([0-9]+))\\.(?P<minor>([0-9]+))\\.(?P<patch>([0-9]+)).*')
            m = Z3VERSION.match(received_version.decode('utf-8'))
            (major, minor, patch) = map(int, (m.group('major'), m.group('minor'), m.group('patch')))
            parsed_version = Version(major, minor, patch)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse Z3 version: '{str(received_version)}'. Assuming compatibility.")
            parsed_version = Version(float('inf'), float('inf'), float('inf'))
        return parsed_version

class YicesSolver(SMTLIBSolver):
    sname = 'yices'

    @classmethod
    def command(self) -> str:
        if False:
            while True:
                i = 10
        return f'{consts.yices_bin} --timeout={consts.timeout}  --incremental'

    @classmethod
    def inits(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return ['(set-logic QF_AUFBV)']

    def __init__(self):
        if False:
            print('Hello World!')
        init = self.inits()
        command = self.command()
        self.ncores = 1
        super().__init__(command=command, init=init, debug=False, support_minmax=False, support_reset=False)

class CVC4Solver(SMTLIBSolver):
    sname = 'cvc4'

    @classmethod
    def command(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{consts.cvc4_bin} --tlimit={consts.timeout * 1000} --lang=smt2 --incremental'

    @classmethod
    def inits(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return ['(set-logic QF_AUFBV)', '(set-option :produce-models true)']

    def __init__(self):
        if False:
            while True:
                i = 10
        init = self.inits()
        command = self.command()
        self.ncores = 1
        super().__init__(command=command, init=init)

class BoolectorSolver(SMTLIBSolver):
    sname = 'boolector'

    @classmethod
    def command(self) -> str:
        if False:
            while True:
                i = 10
        return f'{consts.boolector_bin} --time={consts.timeout} -i'

    @classmethod
    def inits(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return ['(set-logic QF_AUFBV)', '(set-option :produce-models true)']

    def __init__(self, args: List[str]=[]):
        if False:
            return 10
        init = self.inits()
        command = self.command()
        self.ncores = 1
        super().__init__(command=command, init=init)

class SmtlibPortfolio:

    def __init__(self, solvers: List[str], debug: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Single smtlib interactive process\n\n        :param command: the shell command to execute\n        :param debug: log all messaging\n        '
        self._procs: Dict[str, SmtlibProc] = {}
        self._solvers: List[str] = solvers
        self._debug = debug

    def start(self):
        if False:
            return 10
        if len(self._procs) == 0:
            for solver in self._solvers:
                self._procs[solver] = SmtlibProc(solver_selector[solver].command(), self._debug)
        for (_, proc) in self._procs.items():
            proc.start()

    def stop(self):
        if False:
            while True:
                i = 10
        "\n        Stops the solver process by:\n        - sending a SIGKILL signal,\n        - waiting till the process terminates (so we don't leave a zombie process)\n        "
        for (solver, proc) in self._procs.items():
            proc.stop()

    def send(self, cmd: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Send a string to the solver.\n\n        :param cmd: a SMTLIBv2 command (ex. (check-sat))\n        '
        assert len(self._procs) > 0
        inds = list(range(len(self._procs)))
        shuffle(inds)
        for i in inds:
            solver = self._solvers[i]
            proc = self._procs[solver]
            if not proc.is_started():
                continue
            proc.send(cmd)

    def recv(self) -> str:
        if False:
            print('Hello World!')
        'Reads the response from the smtlib solver'
        tries = 0
        timeout = 0.0
        inds = list(range(len(self._procs)))
        while True:
            shuffle(inds)
            for i in inds:
                solver = self._solvers[i]
                proc = self._procs[solver]
                if not proc.is_started():
                    continue
                buf = proc.recv(wait=False)
                if buf is not None:
                    for osolver in self._solvers:
                        if osolver != solver:
                            self._procs[osolver].stop()
                    return buf
                else:
                    tries += 1
            if tries > 10 * len(self._procs):
                time.sleep(timeout)
                timeout += 0.1

    def _restart(self) -> None:
        if False:
            print('Hello World!')
        'Auxiliary to start or restart the external solver'
        self.stop()
        self.start()

    def is_started(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._procs) > 0

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        assert len(self._solvers) == len(self._procs)
        for (solver, proc) in self._procs.items():
            for cfg in solver_selector[solver].inits():
                proc.send(cfg)

class PortfolioSolver(SMTLIBSolver):
    sname = 'portfolio'

    def __init__(self):
        if False:
            while True:
                i = 10
        solvers = []
        if shutil.which(consts.yices_bin):
            solvers.append(consts.solver.yices.name)
        if shutil.which(consts.cvc4_bin):
            solvers.append(consts.solver.cvc4.name)
        if shutil.which(consts.boolector_bin):
            solvers.append(consts.solver.boolector.name)
        if solvers == []:
            raise SolverException(f'No Solver not found. Install one ({consts.yices_bin}, {consts.z3_bin}, {consts.cvc4_bin}, {consts.boolector_bin}).')
        logger.info('Creating portfolio with solvers: ' + ','.join(solvers))
        assert len(solvers) > 0
        support_reset: bool = False
        support_minmax: bool = False
        support_pushpop: bool = False
        multiple_check: bool = True
        debug: bool = False
        self._smtlib: SmtlibPortfolio = SmtlibPortfolio(solvers, debug)
        self._support_minmax = support_minmax
        self._support_reset = support_reset
        self._support_pushpop = support_pushpop
        self._multiple_check = multiple_check
        if not self._support_pushpop:
            setattr(self, '_push', None)
            setattr(self, '_pop', None)
        if self._support_minmax and consts.optimize:
            setattr(self, 'optimize', self._optimize_fancy)
        else:
            setattr(self, 'optimize', self._optimize_generic)
        self.ncores = len(solvers)

    def _reset(self, constraints: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Auxiliary method to reset the smtlib external solver to initial defaults'
        if self._support_reset:
            self._smtlib.start()
            self._smtlib.send('(reset)')
        else:
            self._smtlib.stop()
            self._smtlib.start()
        self._smtlib.init()
        if constraints is not None:
            self._smtlib.send(constraints)
solver_selector = {'cvc4': CVC4Solver, 'boolector': BoolectorSolver, 'yices': YicesSolver, 'z3': Z3Solver, 'portfolio': PortfolioSolver}

class SelectedSolver:
    choice = None

    @classmethod
    def instance(cls):
        if False:
            while True:
                i = 10
        if consts.solver == consts.solver.auto:
            if cls.choice is None:
                if shutil.which(consts.yices_bin):
                    cls.choice = consts.solver.yices
                elif shutil.which(consts.z3_bin):
                    cls.choice = consts.solver.z3
                elif shutil.which(consts.cvc4_bin):
                    cls.choice = consts.solver.cvc4
                elif shutil.which(consts.boolector_bin):
                    cls.choice = consts.solver.boolector
                else:
                    raise SolverException(f'No Solver not found. Install one ({consts.yices_bin}, {consts.z3_bin}, {consts.cvc4_bin}, {consts.boolector_bin}).')
        else:
            cls.choice = consts.solver
        SelectedSolver = solver_selector[cls.choice.name]
        return SelectedSolver.instance()