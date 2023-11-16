import functools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
import pytest
from memray import AllocatorType
from memray import FileReader
from memray import Tracker
from memray._test import MemoryAllocator
from tests.utils import filter_relevant_allocations
HERE = Path(__file__).parent
TEST_MULTITHREADED_EXTENSION = HERE / 'multithreaded_extension'
TEST_NATIVE_EXTENSION = HERE / 'native_extension'

def test_multithreaded_extension_with_native_tracking(tmpdir, monkeypatch):
    if False:
        i = 10
        return i + 15
    'Test tracking allocations in a native extension which spawns multiple threads,\n    each thread allocating and freeing memory.'
    output = Path(tmpdir) / 'test.bin'
    extension_name = 'multithreaded_extension'
    extension_path = tmpdir / extension_name
    shutil.copytree(TEST_MULTITHREADED_EXTENSION, extension_path)
    subprocess.run([sys.executable, str(extension_path / 'setup.py'), 'build_ext', '--inplace'], check=True, cwd=extension_path, capture_output=True)
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, 'path', [*sys.path, str(extension_path)])
        from testext import run
        with Tracker(output, native_traces=True):
            run()
    records = list(FileReader(output).get_allocation_records())
    memaligns = []
    memalign_frees = []
    outstanding_memaligns = set()
    for record in records:
        if record.allocator == AllocatorType.POSIX_MEMALIGN:
            memaligns.append(record)
            outstanding_memaligns.add(record.address)
        elif record.allocator == AllocatorType.FREE:
            if record.address in outstanding_memaligns:
                outstanding_memaligns.remove(record.address)
                memalign_frees.append(record)
    assert len(memaligns) == 100 * 100
    assert all((len(memalign.stack_trace()) == 0 for memalign in memaligns))
    expected_symbols = ['allocate_memory', 'worker']
    assert all((expected_symbols == [stack[0] for stack in record.native_stack_trace()][:2] for record in memaligns))
    assert len(memalign_frees) == 100 * 100
    for record in memalign_frees:
        with pytest.raises(NotImplementedError):
            record.stack_trace()
        with pytest.raises(NotImplementedError):
            record.native_stack_trace()

@pytest.mark.valgrind
def test_simple_call_chain_with_native_tracking(tmpdir, monkeypatch):
    if False:
        i = 10
        return i + 15
    output = Path(tmpdir) / 'test.bin'
    extension_name = 'multithreaded_extension'
    extension_path = tmpdir / extension_name
    shutil.copytree(TEST_NATIVE_EXTENSION, extension_path)
    subprocess.run([sys.executable, str(extension_path / 'setup.py'), 'build_ext', '--inplace'], check=True, cwd=extension_path, capture_output=True)
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, 'path', [*sys.path, str(extension_path)])
        from native_ext import run_simple
        with Tracker(output, native_traces=True):
            run_simple()
    records = list(FileReader(output).get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    (python_stack_trace,) = valloc.stack_trace()
    (func, filename, line) = python_stack_trace
    assert func == 'test_simple_call_chain_with_native_tracking'
    assert filename.endswith(__file__)
    expected_symbols = ['baz', 'bar', 'foo']
    assert expected_symbols == [stack[0] for stack in valloc.native_stack_trace()[:3]]

@pytest.mark.skipif(sys.platform == 'darwin', reason='we cannot use debug information to resolve inline functions on macOS')
def test_inlined_call_chain_with_native_tracking(tmpdir, monkeypatch):
    if False:
        i = 10
        return i + 15
    output = Path(tmpdir) / 'test.bin'
    extension_name = 'multithreaded_extension'
    extension_path = tmpdir / extension_name
    shutil.copytree(TEST_NATIVE_EXTENSION, extension_path)
    subprocess.run([sys.executable, str(extension_path / 'setup.py'), 'build_ext', '--inplace'], check=True, cwd=extension_path, capture_output=True)
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, 'path', [*sys.path, str(extension_path)])
        from native_ext import run_inline
        with Tracker(output, native_traces=True):
            run_inline()
    records = list(FileReader(output).get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    (python_stack_trace,) = valloc.stack_trace()
    (func, filename, line) = python_stack_trace
    assert func == 'test_inlined_call_chain_with_native_tracking'
    assert filename.endswith(__file__)
    expected_symbols = ['baz_inline', 'bar_inline', 'foo_inline']
    assert expected_symbols == [stack[0] for stack in valloc.native_stack_trace()[:3]]

@pytest.mark.valgrind
def test_deep_call_chain_with_native_tracking(tmpdir, monkeypatch):
    if False:
        print('Hello World!')
    output = Path(tmpdir) / 'test.bin'
    extension_name = 'multithreaded_extension'
    extension_path = tmpdir / extension_name
    shutil.copytree(TEST_NATIVE_EXTENSION, extension_path)
    subprocess.run([sys.executable, str(extension_path / 'setup.py'), 'build_ext', '--inplace'], check=True, cwd=extension_path, capture_output=True)
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, 'path', [*sys.path, str(extension_path)])
        from native_ext import run_deep
        with Tracker(output, native_traces=True):
            run_deep(2048)
    records = list(FileReader(output).get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    (python_stack_trace,) = valloc.stack_trace()
    (func, filename, line) = python_stack_trace
    assert func == 'test_deep_call_chain_with_native_tracking'
    assert filename.endswith(__file__)
    expected_symbols = ['baz', 'bar', 'foo']
    native_stack = tuple(valloc.native_stack_trace())
    assert len(native_stack) > 2048
    assert expected_symbols == [stack[0] for stack in native_stack[:3]]
    assert all(('deep_call' in stack[0] for stack in native_stack[3:3 + 2048]))

def test_hybrid_stack_in_pure_python(tmpdir):
    if False:
        return 10
    allocator = MemoryAllocator()
    output = Path(tmpdir) / 'test.bin'
    MAX_RECURSIONS = 4

    def recursive_func(n):
        if False:
            print('Hello World!')
        if n == 1:
            return allocator.valloc(1234)
        return recursive_func(n - 1)
    with Tracker(output, native_traces=True):
        recursive_func(MAX_RECURSIONS)
    records = list(FileReader(output).get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    hybrid_stack = tuple((frame[0] for frame in valloc.hybrid_stack_trace()))
    assert hybrid_stack.count('recursive_func') == MAX_RECURSIONS
    assert hybrid_stack.count('recursive_func') == len(valloc.stack_trace()) - 2 == MAX_RECURSIONS
    assert len(valloc.stack_trace()) <= len(hybrid_stack)
    if sys.version_info < (3, 11):
        assert len(hybrid_stack) <= len(valloc.native_stack_trace())
    assert hybrid_stack[-1] == 'test_hybrid_stack_in_pure_python'

def test_hybrid_stack_in_pure_python_with_callbacks(tmpdir):
    if False:
        print('Hello World!')
    allocator = MemoryAllocator()
    output = Path(tmpdir) / 'test.bin'

    def ham():
        if False:
            i = 10
            return i + 15
        spam()

    def spam():
        if False:
            i = 10
            return i + 15
        functools.partial(foo)()

    def foo():
        if False:
            i = 10
            return i + 15
        bar()

    def bar():
        if False:
            print('Hello World!')
        baz()

    def baz():
        if False:
            return 10
        return allocator.valloc(1234)
    funcs = ('ham', 'spam', 'foo', 'bar', 'baz')
    with Tracker(output, native_traces=True):
        ham()
    records = list(FileReader(output).get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    hybrid_stack = tuple((frame[0] for frame in valloc.hybrid_stack_trace()))
    pos = {func: hybrid_stack.index(func) for func in funcs}
    assert pos['ham'] > pos['spam'] > pos['foo'] > pos['bar'] > pos['baz']
    if sys.version_info >= (3, 11) and sys.implementation.name == 'cpython':
        assert pos['ham'] == pos['spam'] + 1
        assert pos['spam'] > pos['foo'] + 1
        assert pos['foo'] == pos['bar'] + 1
        assert pos['bar'] == pos['baz'] + 1
    else:
        assert pos['ham'] > pos['spam'] + 1
        assert pos['spam'] > pos['foo'] + 1
        assert pos['foo'] > pos['bar'] + 1
        assert pos['bar'] > pos['baz'] + 1
    assert hybrid_stack.count('valloc') == 1
    assert [frame[0] for frame in valloc.stack_trace()].count('valloc') == 1

def test_hybrid_stack_of_allocations_inside_ceval(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    output = Path(tmpdir) / 'test.bin'
    extension_name = 'native_extension'
    extension_path = tmpdir / extension_name
    shutil.copytree(TEST_NATIVE_EXTENSION, extension_path)
    subprocess.run([sys.executable, str(extension_path / 'setup.py'), 'build_ext', '--inplace'], check=True, cwd=extension_path, capture_output=True)
    program = textwrap.dedent('\n        import functools\n        import sys\n\n        import memray\n        import native_ext\n\n\n        def foo():\n            native_ext.run_recursive(1, bar)\n\n\n        def bar(_):\n            pass\n\n\n        with memray.Tracker(sys.argv[1], native_traces=True):\n            functools.partial(foo)()\n        ')
    env = os.environ.copy()
    env['PYTHONMALLOC'] = 'malloc'
    env['PYTHONPATH'] = str(extension_path)
    subprocess.run([sys.executable, '-c', program, str(output)], check=True, env=env)
    records = list(FileReader(output).get_allocation_records())
    found_an_interesting_stack = False
    for record in records:
        try:
            stack = [frame[0] for frame in record.hybrid_stack_trace()]
        except NotImplementedError:
            continue
        print(stack)
        assert 'bar' not in stack
        if 'run_recursive' in stack:
            found_an_interesting_stack = True
            assert stack.index('foo') > stack.index('run_recursive')
    assert found_an_interesting_stack

def test_hybrid_stack_in_recursive_python_c_call(tmpdir, monkeypatch):
    if False:
        return 10
    output = Path(tmpdir) / 'test.bin'
    extension_name = 'multithreaded_extension'
    extension_path = tmpdir / extension_name
    shutil.copytree(TEST_NATIVE_EXTENSION, extension_path)
    subprocess.run([sys.executable, str(extension_path / 'setup.py'), 'build_ext', '--inplace'], check=True, cwd=extension_path, capture_output=True)
    MAX_RECURSIONS = 4
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, 'path', [*sys.path, str(extension_path)])
        from native_ext import run_recursive

        def callback(n):
            if False:
                return 10
            return run_recursive(n, callback)
        with Tracker(output, native_traces=True):
            run_recursive(MAX_RECURSIONS, callback)
    records = list(FileReader(output).get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    hybrid_stack = tuple((frame[0] for frame in valloc.hybrid_stack_trace()))
    assert hybrid_stack.count('callback') == MAX_RECURSIONS
    assert sum((1 if 'run_recursive' in elem else 0 for elem in hybrid_stack)) == MAX_RECURSIONS + 1
    assert hybrid_stack.count('callback') == MAX_RECURSIONS
    assert len(valloc.stack_trace()) == MAX_RECURSIONS + 1
    assert valloc.stack_trace()[-1][0] == 'test_hybrid_stack_in_recursive_python_c_call'
    assert len(valloc.stack_trace()) <= len(hybrid_stack) <= len(valloc.native_stack_trace())
    assert hybrid_stack[-1] == 'test_hybrid_stack_in_recursive_python_c_call'

def test_hybrid_stack_in_a_thread(tmpdir, monkeypatch):
    if False:
        while True:
            i = 10
    output = Path(tmpdir) / 'test.bin'
    extension_name = 'multithreaded_extension'
    extension_path = tmpdir / extension_name
    shutil.copytree(TEST_NATIVE_EXTENSION, extension_path)
    subprocess.run([sys.executable, str(extension_path / 'setup.py'), 'build_ext', '--inplace'], check=True, cwd=extension_path, capture_output=True)
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, 'path', [*sys.path, str(extension_path)])
        from native_ext import run_in_thread
        with Tracker(output, native_traces=True):
            run_in_thread()
    records = list(FileReader(output).get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]
    assert len(vallocs) == 1
    (valloc,) = vallocs
    assert len(valloc.stack_trace()) == 0
    expected_symbols = ['baz', 'bar', 'foo']
    assert expected_symbols == [stack[0] for stack in valloc.hybrid_stack_trace()][:3]

def test_hybrid_stack_of_python_thread_starts_with_native_frames(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    "Ensure there are native frames above a thread's first Python frame."
    allocator = MemoryAllocator()
    output = tmp_path / 'test.bin'

    def func():
        if False:
            i = 10
            return i + 15
        allocator.valloc(1234)
        allocator.free()
    with Tracker(output, native_traces=True):
        thread = threading.Thread(target=func)
        thread.start()
        thread.join()
    allocations = list(FileReader(output).get_allocation_records())
    vallocs = [event for event in allocations if event.size == 1234 and event.allocator == AllocatorType.VALLOC]
    (valloc,) = vallocs
    assert not valloc.hybrid_stack_trace()[-1][1].endswith('.py')

@pytest.mark.parametrize('native_traces', [True, False])
def test_native_tracing_header(native_traces, tmpdir):
    if False:
        i = 10
        return i + 15
    allocator = MemoryAllocator()
    output = Path(tmpdir) / 'test.bin'
    with Tracker(output, native_traces=native_traces):
        allocator.valloc(1234)
    assert FileReader(output).metadata.has_native_traces is native_traces