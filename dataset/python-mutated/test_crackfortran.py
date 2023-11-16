import importlib
import codecs
import time
import unicodedata
import pytest
import numpy as np
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern
from . import util
from numpy.f2py import crackfortran
import textwrap
import contextlib
import io

class TestNoSpace(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'gh15035.f')]

    def test_module(self):
        if False:
            i = 10
            return i + 15
        k = np.array([1, 2, 3], dtype=np.float64)
        w = np.array([1, 2, 3], dtype=np.float64)
        self.module.subb(k)
        assert np.allclose(k, w + 1)
        self.module.subc([w, k])
        assert np.allclose(k, w + 1)
        assert self.module.t0('23') == b'2'

class TestPublicPrivate:

    def test_defaultPrivate(self):
        if False:
            for i in range(10):
                print('nop')
        fpath = util.getpath('tests', 'src', 'crackfortran', 'privatemod.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert 'private' in mod['vars']['a']['attrspec']
        assert 'public' not in mod['vars']['a']['attrspec']
        assert 'private' in mod['vars']['b']['attrspec']
        assert 'public' not in mod['vars']['b']['attrspec']
        assert 'private' not in mod['vars']['seta']['attrspec']
        assert 'public' in mod['vars']['seta']['attrspec']

    def test_defaultPublic(self, tmp_path):
        if False:
            return 10
        fpath = util.getpath('tests', 'src', 'crackfortran', 'publicmod.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert 'private' in mod['vars']['a']['attrspec']
        assert 'public' not in mod['vars']['a']['attrspec']
        assert 'private' not in mod['vars']['seta']['attrspec']
        assert 'public' in mod['vars']['seta']['attrspec']

    def test_access_type(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        fpath = util.getpath('tests', 'src', 'crackfortran', 'accesstype.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        tt = mod[0]['vars']
        assert set(tt['a']['attrspec']) == {'private', 'bind(c)'}
        assert set(tt['b_']['attrspec']) == {'public', 'bind(c)'}
        assert set(tt['c']['attrspec']) == {'public'}

    def test_nowrap_private_proceedures(self, tmp_path):
        if False:
            while True:
                i = 10
        fpath = util.getpath('tests', 'src', 'crackfortran', 'gh23879.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        pyf = crackfortran.crack2fortran(mod)
        assert 'bar' not in pyf

class TestModuleProcedure:

    def test_moduleOperators(self, tmp_path):
        if False:
            print('Hello World!')
        fpath = util.getpath('tests', 'src', 'crackfortran', 'operators.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert 'body' in mod and len(mod['body']) == 9
        assert mod['body'][1]['name'] == 'operator(.item.)'
        assert 'implementedby' in mod['body'][1]
        assert mod['body'][1]['implementedby'] == ['item_int', 'item_real']
        assert mod['body'][2]['name'] == 'operator(==)'
        assert 'implementedby' in mod['body'][2]
        assert mod['body'][2]['implementedby'] == ['items_are_equal']
        assert mod['body'][3]['name'] == 'assignment(=)'
        assert 'implementedby' in mod['body'][3]
        assert mod['body'][3]['implementedby'] == ['get_int', 'get_real']

    def test_notPublicPrivate(self, tmp_path):
        if False:
            i = 10
            return i + 15
        fpath = util.getpath('tests', 'src', 'crackfortran', 'pubprivmod.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        mod = mod[0]
        assert mod['vars']['a']['attrspec'] == ['private']
        assert mod['vars']['b']['attrspec'] == ['public']
        assert mod['vars']['seta']['attrspec'] == ['public']

class TestExternal(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'gh17859.f')]

    def test_external_as_statement(self):
        if False:
            while True:
                i = 10

        def incr(x):
            if False:
                while True:
                    i = 10
            return x + 123
        r = self.module.external_as_statement(incr)
        assert r == 123

    def test_external_as_attribute(self):
        if False:
            print('Hello World!')

        def incr(x):
            if False:
                print('Hello World!')
            return x + 123
        r = self.module.external_as_attribute(incr)
        assert r == 123

class TestCrackFortran(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'gh2848.f90')]

    def test_gh2848(self):
        if False:
            i = 10
            return i + 15
        r = self.module.gh2848(1, 2)
        assert r == (1, 2)

class TestMarkinnerspaces:

    def test_do_not_touch_normal_spaces(self):
        if False:
            return 10
        test_list = ['a ', ' a', 'a b c', "'abcdefghij'"]
        for i in test_list:
            assert markinnerspaces(i) == i

    def test_one_relevant_space(self):
        if False:
            print('Hello World!')
        assert markinnerspaces("a 'b c' \\' \\'") == "a 'b@_@c' \\' \\'"
        assert markinnerspaces('a "b c" \\" \\"') == 'a "b@_@c" \\" \\"'

    def test_ignore_inner_quotes(self):
        if False:
            for i in range(10):
                print('nop')
        assert markinnerspaces('a \'b c" " d\' e') == 'a \'b@_@c"@_@"@_@d\' e'
        assert markinnerspaces('a "b c\' \' d" e') == 'a "b@_@c\'@_@\'@_@d" e'

    def test_multiple_relevant_spaces(self):
        if False:
            i = 10
            return i + 15
        assert markinnerspaces("a 'b c' 'd e'") == "a 'b@_@c' 'd@_@e'"
        assert markinnerspaces('a "b c" "d e"') == 'a "b@_@c" "d@_@e"'

class TestDimSpec(util.F2PyTest):
    """This test suite tests various expressions that are used as dimension
    specifications.

    There exists two usage cases where analyzing dimensions
    specifications are important.

    In the first case, the size of output arrays must be defined based
    on the inputs to a Fortran function. Because Fortran supports
    arbitrary bases for indexing, for instance, `arr(lower:upper)`,
    f2py has to evaluate an expression `upper - lower + 1` where
    `lower` and `upper` are arbitrary expressions of input parameters.
    The evaluation is performed in C, so f2py has to translate Fortran
    expressions to valid C expressions (an alternative approach is
    that a developer specifies the corresponding C expressions in a
    .pyf file).

    In the second case, when user provides an input array with a given
    size but some hidden parameters used in dimensions specifications
    need to be determined based on the input array size. This is a
    harder problem because f2py has to solve the inverse problem: find
    a parameter `p` such that `upper(p) - lower(p) + 1` equals to the
    size of input array. In the case when this equation cannot be
    solved (e.g. because the input array size is wrong), raise an
    error before calling the Fortran function (that otherwise would
    likely crash Python process when the size of input arrays is
    wrong). f2py currently supports this case only when the equation
    is linear with respect to unknown parameter.

    """
    suffix = '.f90'
    code_template = textwrap.dedent('\n      function get_arr_size_{count}(a, n) result (length)\n        integer, intent(in) :: n\n        integer, dimension({dimspec}), intent(out) :: a\n        integer length\n        length = size(a)\n      end function\n\n      subroutine get_inv_arr_size_{count}(a, n)\n        integer :: n\n        ! the value of n is computed in f2py wrapper\n        !f2py intent(out) n\n        integer, dimension({dimspec}), intent(in) :: a\n      end subroutine\n    ')
    linear_dimspecs = ['n', '2*n', '2:n', 'n/2', '5 - n/2', '3*n:20', 'n*(n+1):n*(n+5)', '2*n, n']
    nonlinear_dimspecs = ['2*n:3*n*n+2*n']
    all_dimspecs = linear_dimspecs + nonlinear_dimspecs
    code = ''
    for (count, dimspec) in enumerate(all_dimspecs):
        lst = [d.split(':')[0] if ':' in d else '1' for d in dimspec.split(',')]
        code += code_template.format(count=count, dimspec=dimspec, first=', '.join(lst))

    @pytest.mark.parametrize('dimspec', all_dimspecs)
    def test_array_size(self, dimspec):
        if False:
            for i in range(10):
                print('nop')
        count = self.all_dimspecs.index(dimspec)
        get_arr_size = getattr(self.module, f'get_arr_size_{count}')
        for n in [1, 2, 3, 4, 5]:
            (sz, a) = get_arr_size(n)
            assert a.size == sz

    @pytest.mark.parametrize('dimspec', all_dimspecs)
    def test_inv_array_size(self, dimspec):
        if False:
            for i in range(10):
                print('nop')
        count = self.all_dimspecs.index(dimspec)
        get_arr_size = getattr(self.module, f'get_arr_size_{count}')
        get_inv_arr_size = getattr(self.module, f'get_inv_arr_size_{count}')
        for n in [1, 2, 3, 4, 5]:
            (sz, a) = get_arr_size(n)
            if dimspec in self.nonlinear_dimspecs:
                n1 = get_inv_arr_size(a, n)
            else:
                n1 = get_inv_arr_size(a)
            (sz1, _) = get_arr_size(n1)
            assert sz == sz1, (n, n1, sz, sz1)

class TestModuleDeclaration:

    def test_dependencies(self, tmp_path):
        if False:
            i = 10
            return i + 15
        fpath = util.getpath('tests', 'src', 'crackfortran', 'foo_deps.f90')
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        assert mod[0]['vars']['abar']['='] == "bar('abar')"

class TestEval(util.F2PyTest):

    def test_eval_scalar(self):
        if False:
            while True:
                i = 10
        eval_scalar = crackfortran._eval_scalar
        assert eval_scalar('123', {}) == '123'
        assert eval_scalar('12 + 3', {}) == '15'
        assert eval_scalar('a + b', dict(a=1, b=2)) == '3'
        assert eval_scalar('"123"', {}) == "'123'"

class TestFortranReader(util.F2PyTest):

    @pytest.mark.parametrize('encoding', ['ascii', 'utf-8', 'utf-16', 'utf-32'])
    def test_input_encoding(self, tmp_path, encoding):
        if False:
            while True:
                i = 10
        f_path = tmp_path / f'input_with_{encoding}_encoding.f90'
        with f_path.open('w', encoding=encoding) as ff:
            ff.write('\n                     subroutine foo()\n                     end subroutine foo\n                     ')
        mod = crackfortran.crackfortran([str(f_path)])
        assert mod[0]['name'] == 'foo'

class TestUnicodeComment(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'unicode_comment.f90')]

    @pytest.mark.skipif(importlib.util.find_spec('charset_normalizer') is None, reason='test requires charset_normalizer which is not installed')
    def test_encoding_comment(self):
        if False:
            return 10
        self.module.foo(3)

class TestNameArgsPatternBacktracking:

    @pytest.mark.parametrize(['adversary'], [('@)@bind@(@',), ('@)@bind                         @(@',), ('@)@bind foo bar baz@(@',)])
    def test_nameargspattern_backtracking(self, adversary):
        if False:
            print('Hello World!')
        'address ReDOS vulnerability:\n        https://github.com/numpy/numpy/issues/23338'
        trials_per_batch = 12
        batches_per_regex = 4
        (start_reps, end_reps) = (15, 25)
        for ii in range(start_reps, end_reps):
            repeated_adversary = adversary * ii
            for _ in range(batches_per_regex):
                times = []
                for _ in range(trials_per_batch):
                    t0 = time.perf_counter()
                    mtch = nameargspattern.search(repeated_adversary)
                    times.append(time.perf_counter() - t0)
                assert np.median(times) < 0.2
            assert not mtch
            good_version_of_adversary = repeated_adversary + '@)@'
            assert nameargspattern.search(good_version_of_adversary)

class TestFunctionReturn(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'crackfortran', 'gh23598.f90')]

    def test_function_rettype(self):
        if False:
            return 10
        assert self.module.intproduct(3, 4) == 12

class TestFortranGroupCounters(util.F2PyTest):

    def test_end_if_comment(self):
        if False:
            for i in range(10):
                print('nop')
        fpath = util.getpath('tests', 'src', 'crackfortran', 'gh23533.f')
        try:
            crackfortran.crackfortran([str(fpath)])
        except Exception as exc:
            assert False, f"'crackfortran.crackfortran' raised an exception {exc}"

class TestF77CommonBlockReader:

    def test_gh22648(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        fpath = util.getpath('tests', 'src', 'crackfortran', 'gh22648.pyf')
        with contextlib.redirect_stdout(io.StringIO()) as stdout_f2py:
            mod = crackfortran.crackfortran([str(fpath)])
        assert 'Mismatch' not in stdout_f2py.getvalue()