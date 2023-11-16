import pathlib
import pytest
import spack.directives
import spack.error
from spack.error import SpecError, UnsatisfiableSpecError
from spack.spec import ArchSpec, CompilerSpec, DependencySpec, Spec, SpecFormatSigilError, SpecFormatStringError, UnsupportedCompilerError
from spack.variant import InvalidVariantValueError, MultipleValuesInExclusiveVariantError, UnknownVariantError

@pytest.mark.usefixtures('config', 'mock_packages')
class TestSpecSemantics:
    """Test satisfies(), intersects(), constrain() and other semantic operations on specs."""

    @pytest.mark.parametrize('lhs,rhs,expected', [('libelf@0.8.13', '@0:1', 'libelf@0.8.13'), ('libdwarf^libelf@0.8.13', '^libelf@0:1', 'libdwarf^libelf@0.8.13'), ('libelf', Spec(), 'libelf'), ('libdwarf', Spec(), 'libdwarf'), ('%intel', Spec(), '%intel'), ('^mpi', Spec(), '^mpi'), ('+debug', Spec(), '+debug'), ('@3:', Spec(), '@3:'), ('libelf@0:2.5', 'libelf@2.1:3', 'libelf@2.1:2.5'), ('libelf@0:2.5%gcc@2:4.6', 'libelf@2.1:3%gcc@4.5:4.7', 'libelf@2.1:2.5%gcc@4.5:4.6'), ('builtin.mpich', 'mpich', 'builtin.mpich'), ('builtin.mock.mpich', 'mpich', 'builtin.mock.mpich'), ('builtin.mpich', 'builtin.mpich', 'builtin.mpich'), ('mpileaks ^builtin.mock.mpich', '^mpich', 'mpileaks ^builtin.mock.mpich'), ('mpileaks ^builtin.mock.mpich', '^mpi', 'mpileaks ^mpi ^builtin.mock.mpich'), ('mpileaks ^builtin.mock.mpich', '^builtin.mock.mpich', 'mpileaks ^builtin.mock.mpich'), ('foo%gcc', '%gcc', 'foo%gcc'), ('foo%intel', '%intel', 'foo%intel'), ('foo%gcc', '%gcc@4.7.2', 'foo%gcc@4.7.2'), ('foo%intel', '%intel@4.7.2', 'foo%intel@4.7.2'), ('foo%pgi@4.5', '%pgi@4.4:4.6', 'foo%pgi@4.5'), ('foo@2.0%pgi@4.5', '@1:3%pgi@4.4:4.6', 'foo@2.0%pgi@4.5'), ('foo %gcc@4.7.3', '%gcc@4.7', 'foo %gcc@4.7.3'), ('libelf %gcc@4.4.7', 'libelf %gcc@4.4.7', 'libelf %gcc@4.4.7'), ('libelf', 'libelf %gcc@4.4.7', 'libelf %gcc@4.4.7'), ('foo platform=test', 'platform=test', 'foo platform=test'), ('foo platform=linux', 'platform=linux', 'foo platform=linux'), ('foo platform=test', 'platform=test target=frontend', 'foo platform=test target=frontend'), ('foo platform=test', 'platform=test os=frontend target=frontend', 'foo platform=test os=frontend target=frontend'), ('foo platform=test os=frontend target=frontend', 'platform=test', 'foo platform=test os=frontend target=frontend'), ('foo arch=test-None-None', 'platform=test', 'foo platform=test'), ('foo arch=test-None-frontend', 'platform=test target=frontend', 'foo platform=test target=frontend'), ('foo arch=test-frontend-frontend', 'platform=test os=frontend target=frontend', 'foo platform=test os=frontend target=frontend'), ('foo arch=test-frontend-frontend', 'platform=test', 'foo platform=test os=frontend target=frontend'), ('foo platform=test target=backend os=backend', 'platform=test target=backend os=backend', 'foo platform=test target=backend os=backend'), ('libelf target=default_target os=default_os', 'libelf target=default_target os=default_os', 'libelf target=default_target os=default_os'), ('mpileaks ^mpich', '^mpich', 'mpileaks ^mpich'), ('mpileaks ^mpich@2.0', '^mpich@1:3', 'mpileaks ^mpich@2.0'), ('mpileaks ^mpich@2.0 ^callpath@1.5', '^mpich@1:3 ^callpath@1.4:1.6', 'mpileaks^mpich@2.0^callpath@1.5'), ('mpileaks ^mpi', '^mpi', 'mpileaks ^mpi'), ('mpileaks ^mpi', '^mpich', 'mpileaks ^mpi ^mpich'), ('mpileaks^mpi@1.5', '^mpi@1.2:1.6', 'mpileaks^mpi@1.5'), ('mpileaks^mpi@2:', '^mpich', 'mpileaks^mpi@2: ^mpich'), ('mpileaks^mpi@2:', '^mpich@3.0.4', 'mpileaks^mpi@2: ^mpich@3.0.4'), ('mpich+foo', 'mpich+foo', 'mpich+foo'), ('mpich++foo', 'mpich++foo', 'mpich++foo'), ('mpich~foo', 'mpich~foo', 'mpich~foo'), ('mpich~~foo', 'mpich~~foo', 'mpich~~foo'), ('mpich foo=1', 'mpich foo=1', 'mpich foo=1'), ('mpich foo==1', 'mpich foo==1', 'mpich foo==1'), ('mpich+foo', 'mpich foo=True', 'mpich+foo'), ('mpich++foo', 'mpich foo=True', 'mpich+foo'), ('mpich foo=true', 'mpich+foo', 'mpich+foo'), ('mpich foo==true', 'mpich++foo', 'mpich+foo'), ('mpich~foo', 'mpich foo=FALSE', 'mpich~foo'), ('mpich~~foo', 'mpich foo=FALSE', 'mpich~foo'), ('mpich foo=False', 'mpich~foo', 'mpich~foo'), ('mpich foo==False', 'mpich~foo', 'mpich~foo'), ('mpich foo=*', 'mpich~foo', 'mpich~foo'), ('mpich+foo', 'mpich foo=*', 'mpich+foo'), ('multivalue-variant foo="bar,baz"', 'multivalue-variant foo=bar,baz', 'multivalue-variant foo=bar,baz'), ('multivalue-variant foo="bar,baz"', 'multivalue-variant foo=*', 'multivalue-variant foo=bar,baz'), ('multivalue-variant foo="bar,baz"', 'multivalue-variant foo=bar', 'multivalue-variant foo=bar,baz'), ('multivalue-variant foo="bar,baz"', 'multivalue-variant foo=baz', 'multivalue-variant foo=bar,baz'), ('multivalue-variant foo="bar,baz,barbaz"', 'multivalue-variant foo=bar,baz', 'multivalue-variant foo=bar,baz,barbaz'), ('multivalue-variant foo="bar,baz"', 'foo="baz,bar"', 'multivalue-variant foo=bar,baz'), ('mpich+foo', 'mpich', 'mpich+foo'), ('mpich~foo', 'mpich', 'mpich~foo'), ('mpich foo=1', 'mpich', 'mpich foo=1'), ('mpich', 'mpich++foo', 'mpich+foo'), ('libelf+debug', 'libelf+foo', 'libelf+debug+foo'), ('libelf+debug', 'libelf+debug+foo', 'libelf+debug+foo'), ('libelf debug=2', 'libelf foo=1', 'libelf debug=2 foo=1'), ('libelf debug=2', 'libelf debug=2 foo=1', 'libelf debug=2 foo=1'), ('libelf+debug', 'libelf~foo', 'libelf+debug~foo'), ('libelf+debug', 'libelf+debug~foo', 'libelf+debug~foo'), ('libelf++debug', 'libelf+debug+foo', 'libelf++debug++foo'), ('libelf debug==2', 'libelf foo=1', 'libelf debug==2 foo==1'), ('libelf debug==2', 'libelf debug=2 foo=1', 'libelf debug==2 foo==1'), ('libelf++debug', 'libelf++debug~foo', 'libelf++debug~~foo'), ('libelf foo=bar,baz', 'libelf foo=*', 'libelf foo=bar,baz'), ('libelf foo=*', 'libelf foo=bar,baz', 'libelf foo=bar,baz'), ('multivalue-variant foo="bar"', 'multivalue-variant foo="baz"', 'multivalue-variant foo="bar,baz"'), ('multivalue-variant foo="bar,barbaz"', 'multivalue-variant foo="baz"', 'multivalue-variant foo="bar,baz,barbaz"'), ('mpich ', 'mpich cppflags="-O3"', 'mpich cppflags="-O3"'), ('mpich cppflags="-O3 -Wall"', 'mpich cppflags="-O3 -Wall"', 'mpich cppflags="-O3 -Wall"'), ('mpich cppflags=="-O3"', 'mpich cppflags=="-O3"', 'mpich cppflags=="-O3"'), ('libelf cflags="-O3"', 'libelf cppflags="-Wall"', 'libelf cflags="-O3" cppflags="-Wall"'), ('libelf cflags="-O3"', 'libelf cppflags=="-Wall"', 'libelf cflags="-O3" cppflags=="-Wall"'), ('libelf cflags=="-O3"', 'libelf cppflags=="-Wall"', 'libelf cflags=="-O3" cppflags=="-Wall"'), ('libelf cflags="-O3"', 'libelf cflags="-O3" cppflags="-Wall"', 'libelf cflags="-O3" cppflags="-Wall"')])
    def test_abstract_specs_can_constrain_each_other(self, lhs, rhs, expected):
        if False:
            print('Hello World!')
        'Test that lhs and rhs intersect with each other, and that they can be constrained\n        with each other. Also check that the constrained result match the expected spec.\n        '
        (lhs, rhs, expected) = (Spec(lhs), Spec(rhs), Spec(expected))
        assert lhs.intersects(rhs)
        assert rhs.intersects(lhs)
        (c1, c2) = (lhs.copy(), rhs.copy())
        c1.constrain(rhs)
        c2.constrain(lhs)
        assert c1 == c2
        assert c1 == expected

    def test_constrain_specs_by_hash(self, default_mock_concretization, database):
        if False:
            print('Hello World!')
        'Test that Specs specified only by their hashes can constrain eachother.'
        mpich_dag_hash = '/' + database.query_one('mpich').dag_hash()
        spec = Spec(mpich_dag_hash[:7])
        assert spec.constrain(Spec(mpich_dag_hash)) is False
        assert spec.abstract_hash == mpich_dag_hash[1:]

    def test_mismatched_constrain_spec_by_hash(self, default_mock_concretization, database):
        if False:
            i = 10
            return i + 15
        'Test that Specs specified only by their incompatible hashes fail appropriately.'
        lhs = '/' + database.query_one('callpath ^mpich').dag_hash()
        rhs = '/' + database.query_one('callpath ^mpich2').dag_hash()
        with pytest.raises(spack.spec.InvalidHashError):
            Spec(lhs).constrain(Spec(rhs))
        with pytest.raises(spack.spec.InvalidHashError):
            Spec(lhs[:7]).constrain(Spec(rhs))

    @pytest.mark.parametrize('lhs,rhs', [('libelf', Spec()), ('libelf', '@0:1'), ('libelf', '@0:1 %gcc')])
    def test_concrete_specs_which_satisfies_abstract(self, lhs, rhs, default_mock_concretization):
        if False:
            i = 10
            return i + 15
        'Test that constraining an abstract spec by a compatible concrete one makes the\n        abstract spec concrete, and equal to the one it was constrained with.\n        '
        (lhs, rhs) = (default_mock_concretization(lhs), Spec(rhs))
        assert lhs.intersects(rhs)
        assert rhs.intersects(lhs)
        assert lhs.satisfies(rhs)
        assert not rhs.satisfies(lhs)
        assert lhs.constrain(rhs) is False
        assert rhs.constrain(lhs) is True
        assert rhs.concrete
        assert lhs.satisfies(rhs)
        assert rhs.satisfies(lhs)
        assert lhs == rhs

    @pytest.mark.parametrize('lhs,rhs', [('foo platform=linux', 'platform=test os=redhat6 target=x86'), ('foo os=redhat6', 'platform=test os=debian6 target=x86_64'), ('foo target=x86_64', 'platform=test os=redhat6 target=x86'), ('foo arch=test-frontend-frontend', 'platform=test os=frontend target=backend'), ('foo%intel', '%gcc'), ('foo%intel', '%pgi'), ('foo%pgi@4.3', '%pgi@4.4:4.6'), ('foo@4.0%pgi', '@1:3%pgi'), ('foo@4.0%pgi@4.5', '@1:3%pgi@4.4:4.6'), ('builtin.mock.mpich', 'builtin.mpich'), ('mpileaks ^builtin.mock.mpich', '^builtin.mpich'), ('mpileaks^mpich@1.2', '^mpich@2.0'), ('mpileaks^mpich@4.0^callpath@1.5', '^mpich@1:3^callpath@1.4:1.6'), ('mpileaks^mpich@2.0^callpath@1.7', '^mpich@1:3^callpath@1.4:1.6'), ('mpileaks^mpich@4.0^callpath@1.7', '^mpich@1:3^callpath@1.4:1.6'), ('mpileaks^mpi@3', '^mpi@1.2:1.6'), ('mpileaks^mpi@3:', '^mpich2@1.4'), ('mpileaks^mpi@3:', '^mpich2'), ('mpileaks^mpi@3:', '^mpich@1.0'), ('mpich~foo', 'mpich+foo'), ('mpich+foo', 'mpich~foo'), ('mpich foo=True', 'mpich foo=False'), ('mpich~~foo', 'mpich++foo'), ('mpich++foo', 'mpich~~foo'), ('mpich foo==True', 'mpich foo==False'), ('mpich cppflags="-O3"', 'mpich cppflags="-O2"'), ('mpich cppflags="-O3"', 'mpich cppflags=="-O3"'), ('libelf@0:2.0', 'libelf@2.1:3'), ('libelf@0:2.5%gcc@4.8:4.9', 'libelf@2.1:3%gcc@4.5:4.7'), ('libelf+debug', 'libelf~debug'), ('libelf+debug~foo', 'libelf+debug+foo'), ('libelf debug=True', 'libelf debug=False'), ('libelf cppflags="-O3"', 'libelf cppflags="-O2"'), ('libelf platform=test target=be os=be', 'libelf target=fe os=fe')])
    def test_constraining_abstract_specs_with_empty_intersection(self, lhs, rhs):
        if False:
            i = 10
            return i + 15
        'Check that two abstract specs with an empty intersection cannot be constrained\n        with each other.\n        '
        (lhs, rhs) = (Spec(lhs), Spec(rhs))
        assert not lhs.intersects(rhs)
        assert not rhs.intersects(lhs)
        with pytest.raises(UnsatisfiableSpecError):
            lhs.constrain(rhs)
        with pytest.raises(UnsatisfiableSpecError):
            rhs.constrain(lhs)

    @pytest.mark.parametrize('lhs,rhs', [('mpich', 'mpich +foo'), ('mpich', 'mpich~foo'), ('mpich', 'mpich foo=1'), ('mpich', 'mpich++foo'), ('mpich', 'mpich~~foo'), ('mpich', 'mpich foo==1'), ('mpich', 'mpich cflags="-O3"'), ('mpich cflags=-O3', 'mpich cflags="-O3 -Ofast"'), ('mpich cflags=-O2', 'mpich cflags="-O3"'), ('multivalue-variant foo=bar', 'multivalue-variant +foo'), ('multivalue-variant foo=bar', 'multivalue-variant ~foo'), ('multivalue-variant fee=bar', 'multivalue-variant fee=baz')])
    def test_concrete_specs_which_do_not_satisfy_abstract(self, lhs, rhs, default_mock_concretization):
        if False:
            print('Hello World!')
        (lhs, rhs) = (default_mock_concretization(lhs), Spec(rhs))
        assert lhs.intersects(rhs) is False
        assert rhs.intersects(lhs) is False
        assert not lhs.satisfies(rhs)
        assert not rhs.satisfies(lhs)
        with pytest.raises(UnsatisfiableSpecError):
            assert lhs.constrain(rhs)
        with pytest.raises(UnsatisfiableSpecError):
            assert rhs.constrain(lhs)

    def test_satisfies_single_valued_variant(self):
        if False:
            print('Hello World!')
        'Tests that the case reported in\n        https://github.com/spack/spack/pull/2386#issuecomment-282147639\n        is handled correctly.\n        '
        a = Spec('a foobar=bar')
        a.concretize()
        assert a.satisfies('foobar=bar')
        assert a.satisfies('foobar=*')
        assert 'foobar=bar' in a
        assert 'foobar==bar' in a
        assert 'foobar=baz' not in a
        assert 'foobar=fee' not in a
        assert 'foo=bar' in a
        assert '^b' in a

    def test_unsatisfied_single_valued_variant(self):
        if False:
            return 10
        a = Spec('a foobar=baz')
        a.concretize()
        assert '^b' not in a
        mv = Spec('multivalue-variant')
        mv.concretize()
        assert 'a@1.0' not in mv

    def test_indirect_unsatisfied_single_valued_variant(self):
        if False:
            i = 10
            return i + 15
        spec = Spec('singlevalue-variant-dependent')
        spec.concretize()
        assert 'a@1.0' not in spec

    def test_unsatisfiable_multi_value_variant(self, default_mock_concretization):
        if False:
            i = 10
            return i + 15
        a = default_mock_concretization('multivalue-variant foo="bar"')
        spec_str = 'multivalue-variant foo="bar,baz"'
        b = Spec(spec_str)
        assert not a.satisfies(b)
        assert not a.satisfies(spec_str)
        with pytest.raises(UnsatisfiableSpecError):
            a.constrain(b)
        a = Spec('multivalue-variant foo="bar"')
        spec_str = 'multivalue-variant foo="bar,baz"'
        b = Spec(spec_str)
        assert a.satisfies(b)
        assert a.satisfies(spec_str)
        assert a.constrain(b)
        a = default_mock_concretization('multivalue-variant foo="bar,baz"')
        spec_str = 'multivalue-variant foo="bar,baz,quux"'
        b = Spec(spec_str)
        assert not a.satisfies(b)
        assert not a.satisfies(spec_str)
        with pytest.raises(UnsatisfiableSpecError):
            a.constrain(b)
        a = Spec('multivalue-variant foo="bar,baz"')
        spec_str = 'multivalue-variant foo="bar,baz,quux"'
        b = Spec(spec_str)
        assert a.intersects(b)
        assert a.intersects(spec_str)
        assert a.constrain(b)
        with pytest.raises(InvalidVariantValueError):
            a.concretize()
        a = Spec('multivalue-variant fee="bar"')
        spec_str = 'multivalue-variant fee="baz"'
        b = Spec(spec_str)
        assert a.intersects(b)
        assert a.intersects(spec_str)
        assert a.constrain(b)
        with pytest.raises(MultipleValuesInExclusiveVariantError):
            a.concretize()

    def test_copy_satisfies_transitive(self):
        if False:
            return 10
        spec = Spec('dttop')
        spec.concretize()
        copy = spec.copy()
        for s in spec.traverse():
            assert s.satisfies(copy[s.name])
            assert copy[s.name].satisfies(s)

    def test_intersects_virtual(self):
        if False:
            for i in range(10):
                print('nop')
        assert Spec('mpich').intersects(Spec('mpi'))
        assert Spec('mpich2').intersects(Spec('mpi'))
        assert Spec('zmpi').intersects(Spec('mpi'))

    def test_intersects_virtual_providers(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that we can always intersect virtual providers from abstract specs.\n        Concretization will give meaning to virtuals, and eventually forbid certain\n        configurations.\n        '
        assert Spec('netlib-lapack ^openblas').intersects('netlib-lapack ^openblas')
        assert Spec('netlib-lapack ^netlib-blas').intersects('netlib-lapack ^openblas')
        assert Spec('netlib-lapack ^openblas').intersects('netlib-lapack ^netlib-blas')
        assert Spec('netlib-lapack ^netlib-blas').intersects('netlib-lapack ^netlib-blas')

    def test_intersectable_concrete_specs_must_have_the_same_hash(self):
        if False:
            i = 10
            return i + 15
        'Ensure that concrete specs are matched *exactly* by hash.'
        s1 = Spec('mpileaks').concretized()
        s2 = s1.copy()
        assert s1.satisfies(s2)
        assert s2.satisfies(s1)
        assert s1.intersects(s2)
        s2._hash = s1.dag_hash()[-1::-1]
        assert not s1.satisfies(s2)
        assert not s2.satisfies(s1)
        assert not s1.intersects(s2)

    def test_self_index(self):
        if False:
            i = 10
            return i + 15
        s = Spec('callpath')
        assert s['callpath'] == s

    def test_dep_index(self):
        if False:
            for i in range(10):
                print('nop')
        s = Spec('callpath')
        s.normalize()
        assert s['callpath'] == s
        assert isinstance(s['dyninst'], Spec)
        assert isinstance(s['libdwarf'], Spec)
        assert isinstance(s['libelf'], Spec)
        assert isinstance(s['mpi'], Spec)
        assert s['dyninst'].name == 'dyninst'
        assert s['libdwarf'].name == 'libdwarf'
        assert s['libelf'].name == 'libelf'
        assert s['mpi'].name == 'mpi'

    def test_spec_contains_deps(self):
        if False:
            for i in range(10):
                print('nop')
        s = Spec('callpath')
        s.normalize()
        assert 'dyninst' in s
        assert 'libdwarf' in s
        assert 'libelf' in s
        assert 'mpi' in s

    @pytest.mark.usefixtures('config')
    def test_virtual_index(self):
        if False:
            print('Hello World!')
        s = Spec('callpath')
        s.concretize()
        s_mpich = Spec('callpath ^mpich')
        s_mpich.concretize()
        s_mpich2 = Spec('callpath ^mpich2')
        s_mpich2.concretize()
        s_zmpi = Spec('callpath ^zmpi')
        s_zmpi.concretize()
        assert s['mpi'].name != 'mpi'
        assert s_mpich['mpi'].name == 'mpich'
        assert s_mpich2['mpi'].name == 'mpich2'
        assert s_zmpi['zmpi'].name == 'zmpi'
        for spec in [s, s_mpich, s_mpich2, s_zmpi]:
            assert 'mpi' in spec

    @pytest.mark.parametrize('lhs,rhs', [('libelf', '@1.0'), ('libelf', '@1.0:5.0'), ('libelf', '%gcc'), ('libelf%gcc', '%gcc@4.5'), ('libelf', '+debug'), ('libelf', 'debug=*'), ('libelf', '~debug'), ('libelf', 'debug=2'), ('libelf', 'cppflags="-O3"'), ('libelf', 'cppflags=="-O3"'), ('libelf^foo', 'libelf^foo@1.0'), ('libelf^foo', 'libelf^foo@1.0:5.0'), ('libelf^foo', 'libelf^foo%gcc'), ('libelf^foo%gcc', 'libelf^foo%gcc@4.5'), ('libelf^foo', 'libelf^foo+debug'), ('libelf^foo', 'libelf^foo~debug'), ('libelf', '^foo')])
    def test_lhs_is_changed_when_constraining(self, lhs, rhs):
        if False:
            while True:
                i = 10
        (lhs, rhs) = (Spec(lhs), Spec(rhs))
        assert lhs.intersects(rhs)
        assert rhs.intersects(lhs)
        assert not lhs.satisfies(rhs)
        assert lhs.constrain(rhs) is True
        assert lhs.satisfies(rhs)

    @pytest.mark.parametrize('lhs,rhs', [('libelf', 'libelf'), ('libelf@1.0', '@1.0'), ('libelf@1.0:5.0', '@1.0:5.0'), ('libelf%gcc', '%gcc'), ('libelf%gcc@4.5', '%gcc@4.5'), ('libelf+debug', '+debug'), ('libelf~debug', '~debug'), ('libelf debug=2', 'debug=2'), ('libelf debug=2', 'debug=*'), ('libelf cppflags="-O3"', 'cppflags="-O3"'), ('libelf cppflags=="-O3"', 'cppflags=="-O3"'), ('libelf^foo@1.0', 'libelf^foo@1.0'), ('libelf^foo@1.0:5.0', 'libelf^foo@1.0:5.0'), ('libelf^foo%gcc', 'libelf^foo%gcc'), ('libelf^foo%gcc@4.5', 'libelf^foo%gcc@4.5'), ('libelf^foo+debug', 'libelf^foo+debug'), ('libelf^foo~debug', 'libelf^foo~debug'), ('libelf^foo cppflags="-O3"', 'libelf^foo cppflags="-O3"')])
    def test_lhs_is_not_changed_when_constraining(self, lhs, rhs):
        if False:
            while True:
                i = 10
        (lhs, rhs) = (Spec(lhs), Spec(rhs))
        assert lhs.intersects(rhs)
        assert rhs.intersects(lhs)
        assert lhs.satisfies(rhs)
        assert lhs.constrain(rhs) is False

    def test_exceptional_paths_for_constructor(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError):
            Spec((1, 2))
        with pytest.raises(ValueError):
            Spec('libelf foo')

    def test_spec_formatting(self, default_mock_concretization):
        if False:
            print('Hello World!')
        spec = default_mock_concretization('multivalue-variant cflags=-O2')
        spec_string = str(spec)
        idx = spec_string.index(' ^')
        assert spec_string[:idx] == spec.format().strip()
        package_segments = [('{NAME}', '', 'name', lambda spec: spec), ('{VERSION}', '', 'version', lambda spec: spec), ('{compiler}', '', 'compiler', lambda spec: spec), ('{compiler_flags}', '', 'compiler_flags', lambda spec: spec), ('{variants}', '', 'variants', lambda spec: spec), ('{architecture}', '', 'architecture', lambda spec: spec), ('{@VERSIONS}', '@', 'versions', lambda spec: spec), ('{%compiler}', '%', 'compiler', lambda spec: spec), ('{arch=architecture}', 'arch=', 'architecture', lambda spec: spec), ('{compiler.name}', '', 'name', lambda spec: spec.compiler), ('{compiler.version}', '', 'version', lambda spec: spec.compiler), ('{%compiler.name}', '%', 'name', lambda spec: spec.compiler), ('{@compiler.version}', '@', 'version', lambda spec: spec.compiler), ('{architecture.platform}', '', 'platform', lambda spec: spec.architecture), ('{architecture.os}', '', 'os', lambda spec: spec.architecture), ('{architecture.target}', '', 'target', lambda spec: spec.architecture), ('{prefix}', '', 'prefix', lambda spec: spec), ('{external}', '', 'external', lambda spec: spec)]
        hash_segments = [('{hash:7}', '', lambda s: s.dag_hash(7)), ('{/hash}', '/', lambda s: '/' + s.dag_hash())]
        other_segments = [('{spack_root}', spack.paths.spack_root), ('{spack_install}', spack.store.STORE.layout.root)]

        def depify(depname, fmt_str, sigil):
            if False:
                return 10
            sig = len(sigil)
            opening = fmt_str[:1 + sig]
            closing = fmt_str[1 + sig:]
            return (spec[depname], opening + f'^{depname}.' + closing)

        def check_prop(check_spec, fmt_str, prop, getter):
            if False:
                return 10
            actual = spec.format(fmt_str)
            expected = getter(check_spec)
            assert actual == str(expected).strip()
        for (named_str, sigil, prop, get_component) in package_segments:
            getter = lambda s: sigil + str(getattr(get_component(s), prop, ''))
            check_prop(spec, named_str, prop, getter)
            (mpi, fmt_str) = depify('mpi', named_str, sigil)
            check_prop(mpi, fmt_str, prop, getter)
        for (named_str, sigil, getter) in hash_segments:
            assert spec.format(named_str) == getter(spec)
            (callpath, fmt_str) = depify('callpath', named_str, sigil)
            assert spec.format(fmt_str) == getter(callpath)
        for (named_str, expected) in other_segments:
            actual = spec.format(named_str)
            assert expected == actual

    def test_spec_formatting_escapes(self, default_mock_concretization):
        if False:
            i = 10
            return i + 15
        spec = default_mock_concretization('multivalue-variant cflags=-O2')
        sigil_mismatches = ['{@name}', '{@version.concrete}', '{%compiler.version}', '{/hashd}', '{arch=architecture.os}']
        for fmt_str in sigil_mismatches:
            with pytest.raises(SpecFormatSigilError):
                spec.format(fmt_str)
        bad_formats = ['{}', 'name}', '\\{name}', '{name', '{name\\}', '{_concrete}', '{dag_hash}', '{foo}', '{+variants.debug}']
        for fmt_str in bad_formats:
            with pytest.raises(SpecFormatStringError):
                spec.format(fmt_str)

    @pytest.mark.regression('9908')
    def test_spec_flags_maintain_order(self):
        if False:
            print('Hello World!')
        spec_str = 'libelf %gcc@11.1.0 os=redhat6'
        for _ in range(3):
            s = Spec(spec_str).concretized()
            assert all((s.compiler_flags[x] == ['-O0', '-g'] for x in ('cflags', 'cxxflags', 'fflags')))

    def test_combination_of_wildcard_or_none(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(spack.variant.InvalidVariantValueCombinationError):
            Spec('multivalue-variant foo=none,bar')
        with pytest.raises(spack.variant.InvalidVariantValueCombinationError):
            Spec('multivalue-variant foo=*,bar')

    def test_errors_in_variant_directive(self):
        if False:
            while True:
                i = 10
        variant = spack.directives.variant.__wrapped__

        class Pkg:
            name = 'PKG'
        fn = variant('patches')
        with pytest.raises(spack.directives.DirectiveError) as exc_info:
            fn(Pkg())
        assert "The name 'patches' is reserved" in str(exc_info.value)
        fn = variant('foo', values=spack.variant.any_combination_of('fee', 'foom'), default='bar')
        with pytest.raises(spack.directives.DirectiveError) as exc_info:
            fn(Pkg())
        assert " it is handled by an attribute of the 'values' argument" in str(exc_info.value)
        fn = variant('foo', default=None)
        with pytest.raises(spack.directives.DirectiveError) as exc_info:
            fn(Pkg())
        assert "either a default was not explicitly set, or 'None' was used" in str(exc_info.value)
        fn = variant('foo', default='')
        with pytest.raises(spack.directives.DirectiveError) as exc_info:
            fn(Pkg())
        assert 'the default cannot be an empty string' in str(exc_info.value)

    def test_abstract_spec_prefix_error(self):
        if False:
            return 10
        spec = Spec('libelf')
        with pytest.raises(SpecError):
            spec.prefix

    def test_forwarding_of_architecture_attributes(self):
        if False:
            return 10
        spec = Spec('libelf target=x86_64').concretized()
        assert 'test' in spec.architecture
        assert 'debian' in spec.architecture
        assert 'x86_64' in spec.architecture
        assert spec.platform == 'test'
        assert spec.os == 'debian6'
        assert spec.target == 'x86_64'
        assert spec.target.family == 'x86_64'
        assert 'avx512' not in spec.target
        assert spec.target < 'broadwell'

    @pytest.mark.parametrize('transitive', [True, False])
    def test_splice(self, transitive, default_mock_concretization):
        if False:
            print('Hello World!')
        spec = default_mock_concretization('splice-t')
        dep = default_mock_concretization('splice-h+foo')
        assert dep.dag_hash() != spec['splice-h'].dag_hash()
        out = spec.splice(dep, transitive)
        assert out.concrete
        for node in spec.traverse():
            assert node.name in out
        out_h_build = out['splice-h'].build_spec
        assert out_h_build.dag_hash() == dep.dag_hash()
        expected_z = dep['splice-z'] if transitive else spec['splice-z']
        assert out['splice-z'].dag_hash() == expected_z.dag_hash()
        assert out['splice-t'].build_spec.dag_hash() == spec['splice-t'].dag_hash()
        assert out.spliced

    @pytest.mark.parametrize('transitive', [True, False])
    def test_splice_with_cached_hashes(self, default_mock_concretization, transitive):
        if False:
            return 10
        spec = default_mock_concretization('splice-t')
        dep = default_mock_concretization('splice-h+foo')
        spec._hash = 'aaaaaa'
        dep._hash = 'bbbbbb'
        spec['splice-h']._hash = 'cccccc'
        spec['splice-z']._hash = 'dddddd'
        dep['splice-z']._hash = 'eeeeee'
        out = spec.splice(dep, transitive=transitive)
        out_z_expected = (dep if transitive else spec)['splice-z']
        assert out.dag_hash() != spec.dag_hash()
        assert (out['splice-h'].dag_hash() == dep.dag_hash()) == transitive
        assert out['splice-z'].dag_hash() == out_z_expected.dag_hash()

    @pytest.mark.parametrize('transitive', [True, False])
    def test_splice_input_unchanged(self, default_mock_concretization, transitive):
        if False:
            i = 10
            return i + 15
        spec = default_mock_concretization('splice-t')
        dep = default_mock_concretization('splice-h+foo')
        orig_spec_hash = spec.dag_hash()
        orig_dep_hash = dep.dag_hash()
        spec.splice(dep, transitive)
        assert spec.dag_hash() == orig_spec_hash
        assert dep.dag_hash() == orig_dep_hash

    @pytest.mark.parametrize('transitive', [True, False])
    def test_splice_subsequent(self, default_mock_concretization, transitive):
        if False:
            print('Hello World!')
        spec = default_mock_concretization('splice-t')
        dep = default_mock_concretization('splice-h+foo')
        out = spec.splice(dep, transitive)
        dep = default_mock_concretization('splice-z+bar')
        out2 = out.splice(dep, transitive)
        assert out2.concrete
        assert out2['splice-z'].dag_hash() != spec['splice-z'].dag_hash()
        assert out2['splice-z'].dag_hash() != out['splice-z'].dag_hash()
        assert out2['splice-t'].build_spec.dag_hash() == spec['splice-t'].dag_hash()
        assert out2.spliced

    @pytest.mark.parametrize('transitive', [True, False])
    def test_splice_dict(self, default_mock_concretization, transitive):
        if False:
            print('Hello World!')
        spec = default_mock_concretization('splice-t')
        dep = default_mock_concretization('splice-h+foo')
        out = spec.splice(dep, transitive)
        assert spec.dag_hash() != dep.dag_hash()
        assert out.dag_hash() != dep.dag_hash()
        assert out.dag_hash() != spec.dag_hash()
        node_list = out.to_dict()['spec']['nodes']
        root_nodes = [n for n in node_list if n['hash'] == out.dag_hash()]
        build_spec_nodes = [n for n in node_list if n['hash'] == spec.dag_hash()]
        assert spec.dag_hash() == out.build_spec.dag_hash()
        assert len(root_nodes) == 1
        assert len(build_spec_nodes) == 1

    @pytest.mark.parametrize('transitive', [True, False])
    def test_splice_dict_roundtrip(self, default_mock_concretization, transitive):
        if False:
            for i in range(10):
                print('nop')
        spec = default_mock_concretization('splice-t')
        dep = default_mock_concretization('splice-h+foo')
        out = spec.splice(dep, transitive)
        assert spec.dag_hash() != dep.dag_hash()
        assert out.dag_hash() != dep.dag_hash()
        assert out.dag_hash() != spec.dag_hash()
        out_rt_spec = Spec.from_dict(out.to_dict())
        assert out_rt_spec.dag_hash() == out.dag_hash()
        out_rt_spec_bld_hash = out_rt_spec.build_spec.dag_hash()
        out_rt_spec_h_bld_hash = out_rt_spec['splice-h'].build_spec.dag_hash()
        out_rt_spec_z_bld_hash = out_rt_spec['splice-z'].build_spec.dag_hash()
        assert spec.dag_hash() == out_rt_spec_bld_hash
        assert out_rt_spec.dag_hash() != out_rt_spec_bld_hash
        assert dep['splice-h'].dag_hash() == out_rt_spec_h_bld_hash
        expected_z_bld_hash = dep['splice-z'].dag_hash() if transitive else spec['splice-z'].dag_hash()
        assert expected_z_bld_hash == out_rt_spec_z_bld_hash

    @pytest.mark.parametrize('spec,constraint,expected_result', [('libelf target=haswell', 'target=broadwell', False), ('libelf target=haswell', 'target=haswell', True), ('libelf target=haswell', 'target=x86_64:', True), ('libelf target=haswell', 'target=:haswell', True), ('libelf target=haswell', 'target=icelake,:nocona', False), ('libelf target=haswell', 'target=haswell,:nocona', True), ('libelf target=haswell', 'target=x86_64', False), ('libelf target=x86_64', 'target=haswell', False)])
    @pytest.mark.regression('13111')
    def test_target_constraints(self, spec, constraint, expected_result):
        if False:
            print('Hello World!')
        s = Spec(spec)
        assert s.intersects(constraint) is expected_result

    @pytest.mark.regression('13124')
    def test_error_message_unknown_variant(self):
        if False:
            i = 10
            return i + 15
        s = Spec('mpileaks +unknown')
        with pytest.raises(UnknownVariantError, match='package has no such'):
            s.concretize()

    @pytest.mark.regression('18527')
    def test_satisfies_dependencies_ordered(self):
        if False:
            i = 10
            return i + 15
        d = Spec('zmpi ^fake')
        s = Spec('mpileaks')
        s._add_dependency(d, depflag=0, virtuals=())
        assert s.satisfies('mpileaks ^zmpi ^fake')

    @pytest.mark.parametrize('transitive', [True, False])
    def test_splice_swap_names(self, default_mock_concretization, transitive):
        if False:
            for i in range(10):
                print('nop')
        spec = default_mock_concretization('splice-t')
        dep = default_mock_concretization('splice-a+foo')
        out = spec.splice(dep, transitive)
        assert dep.name in out
        assert transitive == ('+foo' in out['splice-z'])

    @pytest.mark.parametrize('transitive', [True, False])
    def test_splice_swap_names_mismatch_virtuals(self, default_mock_concretization, transitive):
        if False:
            while True:
                i = 10
        spec = default_mock_concretization('splice-t')
        dep = default_mock_concretization('splice-vh+foo')
        with pytest.raises(spack.spec.SpliceError, match='will not provide the same virtuals.'):
            spec.splice(dep, transitive)

    def test_spec_override(self):
        if False:
            while True:
                i = 10
        init_spec = Spec('a foo=baz foobar=baz cflags=-O3 cxxflags=-O1')
        change_spec = Spec('a foo=fee cflags=-O2')
        new_spec = Spec.override(init_spec, change_spec)
        new_spec.concretize()
        assert 'foo=fee' in new_spec
        assert 'foo=baz' not in new_spec
        assert 'foobar=baz' in new_spec
        assert new_spec.compiler_flags['cflags'] == ['-O2']
        assert new_spec.compiler_flags['cxxflags'] == ['-O1']

    @pytest.mark.parametrize('spec_str,specs_in_dag', [('hdf5 ^[virtuals=mpi] mpich', [('mpich', 'mpich'), ('mpi', 'mpich')]), ('netlib-scalapack ^mpich ^openblas-with-lapack', [('mpi', 'mpich'), ('lapack', 'openblas-with-lapack'), ('blas', 'openblas-with-lapack')]), ('netlib-scalapack ^[virtuals=mpi] mpich ^openblas-with-lapack', [('mpi', 'mpich'), ('lapack', 'openblas-with-lapack'), ('blas', 'openblas-with-lapack')]), ('netlib-scalapack ^mpich ^[virtuals=lapack] openblas-with-lapack', [('mpi', 'mpich'), ('lapack', 'openblas-with-lapack'), ('blas', 'openblas-with-lapack')]), ('netlib-scalapack ^[virtuals=mpi] mpich ^[virtuals=lapack] openblas-with-lapack', [('mpi', 'mpich'), ('lapack', 'openblas-with-lapack'), ('blas', 'openblas-with-lapack')]), ('netlib-scalapack ^[virtuals=mpi] intel-parallel-studio ^[virtuals=lapack] openblas-with-lapack', [('mpi', 'intel-parallel-studio'), ('lapack', 'openblas-with-lapack'), ('blas', 'openblas-with-lapack')]), ('netlib-scalapack ^[virtuals=mpi] intel-parallel-studio ^openblas-with-lapack', [('mpi', 'intel-parallel-studio'), ('lapack', 'openblas-with-lapack'), ('blas', 'openblas-with-lapack')]), ('netlib-scalapack ^intel-parallel-studio ^[virtuals=lapack] openblas-with-lapack', [('mpi', 'intel-parallel-studio'), ('lapack', 'openblas-with-lapack'), ('blas', 'openblas-with-lapack')]), ('netlib-scalapack ^[virtuals=lapack,blas] openblas-with-lapack', [('lapack', 'openblas-with-lapack'), ('blas', 'openblas-with-lapack')])])
    def test_virtual_deps_bindings(self, default_mock_concretization, spec_str, specs_in_dag):
        if False:
            i = 10
            return i + 15
        if spack.config.get('config:concretizer') == 'original':
            pytest.skip('Use case not supported by the original concretizer')
        s = default_mock_concretization(spec_str)
        for (label, expected) in specs_in_dag:
            assert label in s
            assert s[label].satisfies(expected), label

    @pytest.mark.parametrize('spec_str', ['netlib-scalapack ^[virtuals=blas] intel-parallel-studio ^openblas-with-lapack', 'netlib-scalapack ^[virtuals=lapack] intel-parallel-studio ^openblas'])
    def test_unsatisfiable_virtual_deps_bindings(self, spec_str):
        if False:
            while True:
                i = 10
        if spack.config.get('config:concretizer') == 'original':
            pytest.skip('Use case not supported by the original concretizer')
        with pytest.raises(spack.solver.asp.UnsatisfiableSpecError):
            Spec(spec_str).concretized()

@pytest.mark.parametrize('spec_str,format_str,expected', [('zlib@git.foo/bar', '{name}-{version}', str(pathlib.Path('zlib-git.foo_bar'))), ('zlib@git.foo/bar', '{name}-{version}-{/hash}', None), ('zlib@git.foo/bar', '{name}/{version}', str(pathlib.Path('zlib', 'git.foo_bar'))), ('zlib@{0}=1.0%gcc'.format('a' * 40), '{name}/{version}/{compiler}', str(pathlib.Path('zlib', '{0}_1.0'.format('a' * 40), 'gcc'))), ('zlib@git.foo/bar=1.0%gcc', '{name}/{version}/{compiler}', str(pathlib.Path('zlib', 'git.foo_bar_1.0', 'gcc')))])
def test_spec_format_path(spec_str, format_str, expected):
    if False:
        i = 10
        return i + 15
    _check_spec_format_path(spec_str, format_str, expected)

def _check_spec_format_path(spec_str, format_str, expected, path_ctor=None):
    if False:
        return 10
    spec = Spec(spec_str)
    if not expected:
        with pytest.raises((spack.spec.SpecFormatPathError, spack.spec.SpecFormatStringError)):
            spec.format_path(format_str, _path_ctor=path_ctor)
    else:
        formatted = spec.format_path(format_str, _path_ctor=path_ctor)
        assert formatted == expected

@pytest.mark.parametrize('spec_str,format_str,expected', [('zlib@git.foo/bar', 'C:\\\\installroot\\{name}\\{version}', 'C:\\installroot\\zlib\\git.foo_bar'), ('zlib@git.foo/bar', '\\\\hostname\\sharename\\{name}\\{version}', '\\\\hostname\\sharename\\zlib\\git.foo_bar'), ('zlib@git.foo/bar', '/installroot/{name}/{version}', 'installroot\\zlib\\git.foo_bar')])
def test_spec_format_path_windows(spec_str, format_str, expected):
    if False:
        i = 10
        return i + 15
    _check_spec_format_path(spec_str, format_str, expected, path_ctor=pathlib.PureWindowsPath)

@pytest.mark.parametrize('spec_str,format_str,expected', [('zlib@git.foo/bar', '/installroot/{name}/{version}', '/installroot/zlib/git.foo_bar'), ('zlib@git.foo/bar', '//installroot/{name}/{version}', '//installroot/zlib/git.foo_bar'), ('zlib@git.foo/bar', 'C:\\\\installroot\\package-{name}-{version}', 'C__installrootpackage-zlib-git.foo_bar'), ('zlib@git.foo/bar', 'package\\{name}\\{version}', None)])
def test_spec_format_path_posix(spec_str, format_str, expected):
    if False:
        return 10
    _check_spec_format_path(spec_str, format_str, expected, path_ctor=pathlib.PurePosixPath)

@pytest.mark.regression('3887')
@pytest.mark.parametrize('spec_str', ['py-extension2', 'extension1', 'perl-extension'])
def test_is_extension_after_round_trip_to_dict(config, mock_packages, spec_str):
    if False:
        while True:
            i = 10
    x = Spec(spec_str).concretized()
    y = Spec.from_dict(x.to_dict())
    for d in y.traverse():
        assert x[d.name].package.is_extension == y[d.name].package.is_extension

def test_malformed_spec_dict():
    if False:
        i = 10
        return i + 15
    with pytest.raises(SpecError, match='malformed'):
        Spec.from_dict({'spec': {'_meta': {'version': 2}, 'nodes': [{'dependencies': {'name': 'foo'}}]}})

def test_spec_dict_hashless_dep():
    if False:
        return 10
    with pytest.raises(SpecError, match="Couldn't parse"):
        Spec.from_dict({'spec': {'_meta': {'version': 2}, 'nodes': [{'name': 'foo', 'hash': 'thehash', 'dependencies': [{'name': 'bar'}]}]}})

@pytest.mark.parametrize('specs,expected', [(['+baz', '+bar'], '+baz+bar'), (['@2.0:', '@:5.1', '+bar'], '@2.0:5.1 +bar'), (['^mpich@3.2', '^mpich@:4.0+foo'], '^mpich@3.2 +foo'), (['^mpich@3.2', '^mpi+foo'], '^mpich@3.2 ^mpi+foo')])
def test_merge_abstract_anonymous_specs(specs, expected):
    if False:
        i = 10
        return i + 15
    specs = [Spec(x) for x in specs]
    result = spack.spec.merge_abstract_anonymous_specs(*specs)
    assert result == Spec(expected)

@pytest.mark.parametrize('anonymous,named,expected', [('+plumed', 'gromacs', 'gromacs+plumed'), ('+plumed ^plumed%gcc', 'gromacs', 'gromacs+plumed ^plumed%gcc'), ('+plumed', 'builtin.gromacs', 'builtin.gromacs+plumed')])
def test_merge_anonymous_spec_with_named_spec(anonymous, named, expected):
    if False:
        print('Hello World!')
    s = Spec(anonymous)
    changed = s.constrain(named)
    assert changed
    assert s == Spec(expected)

def test_spec_installed(default_mock_concretization, database):
    if False:
        while True:
            i = 10
    'Test whether Spec.installed works.'
    specs = database.query()
    spec = specs[0]
    assert spec.installed
    assert spec.copy().installed
    spec = Spec('not-a-real-package')
    assert not spec.installed
    spec = default_mock_concretization('a')
    assert not spec.installed

@pytest.mark.regression('30678')
def test_call_dag_hash_on_old_dag_hash_spec(mock_packages, default_mock_concretization):
    if False:
        for i in range(10):
            print('nop')
    a = default_mock_concretization('a')
    dag_hashes = {spec.name: spec.dag_hash() for spec in a.traverse()}
    for spec in a.traverse():
        assert spec.concrete
        spec._package_hash = None
    for spec in a.traverse():
        assert dag_hashes[spec.name] == spec.dag_hash()
        with pytest.raises(ValueError, match='Cannot call package_hash()'):
            spec.package_hash()

@pytest.mark.regression('30861')
def test_concretize_partial_old_dag_hash_spec(mock_packages, config):
    if False:
        print('Hello World!')
    bottom = Spec('dt-diamond-bottom').concretized()
    delattr(bottom, '_package_hash')
    dummy_hash = 'zd4m26eis2wwbvtyfiliar27wkcv3ehk'
    bottom._hash = dummy_hash
    top = Spec('dt-diamond')
    top.add_dependency_edge(bottom, depflag=0, virtuals=())
    top.concretize()
    for spec in top.traverse():
        assert spec.concrete
    assert spec['dt-diamond-bottom'].dag_hash() == dummy_hash
    assert spec['dt-diamond-bottom']._hash == dummy_hash
    assert not getattr(spec['dt-diamond-bottom'], '_package_hash', None)

def test_unsupported_compiler():
    if False:
        print('Hello World!')
    with pytest.raises(UnsupportedCompilerError):
        Spec('gcc%fake-compiler').validate_or_raise()

def test_package_hash_affects_dunder_and_dag_hash(mock_packages, default_mock_concretization):
    if False:
        return 10
    a1 = default_mock_concretization('a')
    a2 = default_mock_concretization('a')
    assert hash(a1) == hash(a2)
    assert a1.dag_hash() == a2.dag_hash()
    assert a1.process_hash() == a2.process_hash()
    a1.clear_cached_hashes()
    a2.clear_cached_hashes()
    new_hash = '00000000000000000000000000000000'
    if new_hash == a1._package_hash:
        new_hash = '11111111111111111111111111111111'
    a1._package_hash = new_hash
    assert hash(a1) != hash(a2)
    assert a1.dag_hash() != a2.dag_hash()
    assert a1.process_hash() != a2.process_hash()

def test_intersects_and_satisfies_on_concretized_spec(default_mock_concretization):
    if False:
        while True:
            i = 10
    'Test that a spec obtained by concretizing an abstract spec, satisfies the abstract spec\n    but not vice-versa.\n    '
    a1 = default_mock_concretization('a@1.0')
    a2 = Spec('a@1.0')
    assert a1.intersects(a2)
    assert a2.intersects(a1)
    assert a1.satisfies(a2)
    assert not a2.satisfies(a1)

@pytest.mark.parametrize('abstract_spec,spec_str', [('v1-provider', 'v1-consumer ^conditional-provider+disable-v1'), ('conditional-provider', 'v1-consumer ^conditional-provider+disable-v1'), ('^v1-provider', 'v1-consumer ^conditional-provider+disable-v1'), ('^conditional-provider', 'v1-consumer ^conditional-provider+disable-v1')])
@pytest.mark.regression('35597')
def test_abstract_provider_in_spec(abstract_spec, spec_str, default_mock_concretization):
    if False:
        for i in range(10):
            print('nop')
    s = default_mock_concretization(spec_str)
    assert abstract_spec in s

@pytest.mark.parametrize('lhs,rhs,expected', [('a', 'a', True), ('a', 'a@1.0', True), ('a@1.0', 'a', False)])
def test_abstract_contains_semantic(lhs, rhs, expected, mock_packages):
    if False:
        return 10
    (s, t) = (Spec(lhs), Spec(rhs))
    result = s in t
    assert result is expected

@pytest.mark.parametrize('factory,lhs_str,rhs_str,results', [(ArchSpec, 'None-ubuntu20.04-None', 'None-None-x86_64', (True, False, False)), (ArchSpec, 'None-ubuntu20.04-None', 'linux-None-x86_64', (True, False, False)), (ArchSpec, 'None-None-x86_64:', 'linux-None-haswell', (True, False, True)), (ArchSpec, 'None-None-x86_64:haswell', 'linux-None-icelake', (False, False, False)), (ArchSpec, 'linux-None-None', 'linux-None-None', (True, True, True)), (ArchSpec, 'darwin-None-None', 'linux-None-None', (False, False, False)), (ArchSpec, 'None-ubuntu20.04-None', 'None-ubuntu20.04-None', (True, True, True)), (ArchSpec, 'None-ubuntu20.04-None', 'None-ubuntu22.04-None', (False, False, False)), (CompilerSpec, 'gcc', 'clang', (False, False, False)), (CompilerSpec, 'gcc', 'gcc@5', (True, False, True)), (CompilerSpec, 'gcc@5', 'gcc@5.3', (True, False, True)), (CompilerSpec, 'gcc@5', 'gcc@5-tag', (True, False, True)), (Spec, 'cppflags=-foo', 'cppflags=-bar', (False, False, False)), (Spec, "cppflags='-bar -foo'", 'cppflags=-bar', (False, False, False)), (Spec, 'cppflags=-foo', 'cppflags=-foo', (True, True, True)), (Spec, 'cppflags=-foo', 'cflags=-foo', (True, False, False)), (Spec, '@0.94h', '@:0.94i', (True, True, False)), (Spec, 'mpi', 'lapack', (True, False, False)), (Spec, 'mpi', 'pkgconfig', (False, False, False))])
def test_intersects_and_satisfies(factory, lhs_str, rhs_str, results):
    if False:
        while True:
            i = 10
    lhs = factory(lhs_str)
    rhs = factory(rhs_str)
    (intersects, lhs_satisfies_rhs, rhs_satisfies_lhs) = results
    assert lhs.intersects(rhs) is intersects
    assert rhs.intersects(lhs) is lhs.intersects(rhs)
    assert lhs.satisfies(rhs) is lhs_satisfies_rhs
    assert rhs.satisfies(lhs) is rhs_satisfies_lhs

@pytest.mark.parametrize('factory,lhs_str,rhs_str,result,constrained_str', [(ArchSpec, 'None-ubuntu20.04-None', 'None-None-x86_64', True, 'None-ubuntu20.04-x86_64'), (ArchSpec, 'None-None-x86_64', 'None-None-x86_64', False, 'None-None-x86_64'), (ArchSpec, 'None-None-x86_64:icelake', 'None-None-x86_64:icelake', False, 'None-None-x86_64:icelake'), (ArchSpec, 'None-ubuntu20.04-None', 'linux-None-x86_64', True, 'linux-ubuntu20.04-x86_64'), (ArchSpec, 'None-ubuntu20.04-nocona:haswell', 'None-None-x86_64:icelake', False, 'None-ubuntu20.04-nocona:haswell'), (ArchSpec, 'None-ubuntu20.04-nocona,haswell', 'None-None-x86_64:icelake', False, 'None-ubuntu20.04-nocona,haswell'), (CompilerSpec, 'gcc@5', 'gcc@5-tag', True, 'gcc@5-tag'), (CompilerSpec, 'gcc@5', 'gcc@5', False, 'gcc@5'), (Spec, 'cppflags=-foo', 'cppflags=-foo', False, 'cppflags=-foo'), (Spec, 'cppflags=-foo', 'cflags=-foo', True, 'cppflags=-foo cflags=-foo')])
def test_constrain(factory, lhs_str, rhs_str, result, constrained_str):
    if False:
        for i in range(10):
            print('nop')
    lhs = factory(lhs_str)
    rhs = factory(rhs_str)
    assert lhs.constrain(rhs) is result
    assert lhs == factory(constrained_str)
    lhs = factory(lhs_str)
    rhs = factory(rhs_str)
    rhs.constrain(lhs)
    assert rhs == factory(constrained_str)

def test_abstract_hash_intersects_and_satisfies(default_mock_concretization):
    if False:
        i = 10
        return i + 15
    concrete: Spec = default_mock_concretization('a')
    hash = concrete.dag_hash()
    hash_5 = hash[:5]
    hash_6 = hash[:6]
    hash_other = f"{('a' if hash_5[0] == 'b' else 'b')}{hash_5[1:]}"
    abstract_5 = Spec(f'a/{hash_5}')
    abstract_6 = Spec(f'a/{hash_6}')
    abstract_none = Spec(f'a/{hash_other}')
    abstract = Spec('a')

    def assert_subset(a: Spec, b: Spec):
        if False:
            print('Hello World!')
        assert a.intersects(b) and b.intersects(a) and a.satisfies(b) and (not b.satisfies(a))

    def assert_disjoint(a: Spec, b: Spec):
        if False:
            for i in range(10):
                print('nop')
        assert not a.intersects(b) and (not b.intersects(a)) and (not a.satisfies(b)) and (not b.satisfies(a))
    assert_subset(concrete, abstract_5)
    assert_subset(abstract_6, abstract_5)
    assert_subset(abstract_5, abstract)
    assert_disjoint(abstract_none, concrete)
    assert_disjoint(abstract_none, abstract_5)

def test_edge_equality_does_not_depend_on_virtual_order():
    if False:
        i = 10
        return i + 15
    'Tests that two edges that are constructed with just a different order of the virtuals in\n    the input parameters are equal to each other.\n    '
    (parent, child) = (Spec('parent'), Spec('child'))
    edge1 = DependencySpec(parent, child, depflag=0, virtuals=('mpi', 'lapack'))
    edge2 = DependencySpec(parent, child, depflag=0, virtuals=('lapack', 'mpi'))
    assert edge1 == edge2
    assert tuple(sorted(edge1.virtuals)) == edge1.virtuals
    assert tuple(sorted(edge2.virtuals)) == edge1.virtuals