import subprocess
import sys
import textwrap
from pathlib import Path
import pytest
from memray import AllocatorType
from memray import FileReader
from tests.utils import filter_relevant_allocations
pytestmark = pytest.mark.skipif(sys.version_info >= (3, 12), reason='Greenlet does not yet support Python 3.12')

def test_integration_with_greenlet(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Verify that we can track Python stacks when greenlet is in use.'
    output = Path(tmpdir) / 'test.bin'
    subprocess_code = textwrap.dedent(f'\n        import greenlet\n\n        from memray import Tracker\n        from memray._test import MemoryAllocator\n\n\n        def apple():\n            banana()\n\n\n        def banana():\n            allocator.valloc(1024 * 10)\n            animal.switch()\n            allocator.valloc(1024 * 30)\n\n\n        def ant():\n            allocator.valloc(1024 * 20)\n            fruit.switch()\n            allocator.valloc(1024 * 40)\n            bat()\n            allocator.valloc(1024 * 60)\n\n\n        def bat():\n            allocator.valloc(1024 * 50)\n\n\n        def test():\n            fruit.switch()\n            assert fruit.dead\n            animal.switch()\n            assert animal.dead\n            allocator.valloc(1024 * 70)\n\n\n        allocator = MemoryAllocator()\n        output = "{output}"\n\n        with Tracker(output):\n            fruit = greenlet.greenlet(apple)\n            animal = greenlet.greenlet(ant)\n            test()\n        ')
    subprocess.run([sys.executable, '-Xdev', '-c', subprocess_code], timeout=5)
    reader = FileReader(output)
    records = list(reader.get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]

    def stack(alloc):
        if False:
            return 10
        return [frame[0] for frame in alloc.stack_trace()]
    assert stack(vallocs[0]) == ['valloc', 'banana', 'apple']
    assert vallocs[0].size == 10 * 1024
    assert stack(vallocs[1]) == ['valloc', 'ant']
    assert vallocs[1].size == 20 * 1024
    assert stack(vallocs[2]) == ['valloc', 'banana', 'apple']
    assert vallocs[2].size == 30 * 1024
    assert stack(vallocs[3]) == ['valloc', 'ant']
    assert vallocs[3].size == 40 * 1024
    assert stack(vallocs[4]) == ['valloc', 'bat', 'ant']
    assert vallocs[4].size == 50 * 1024
    assert stack(vallocs[5]) == ['valloc', 'ant']
    assert vallocs[5].size == 60 * 1024
    assert stack(vallocs[6]) == ['valloc', 'test', '<module>']
    assert vallocs[6].size == 70 * 1024

def test_importing_greenlet_after_tracking_starts(tmpdir):
    if False:
        return 10
    output = Path(tmpdir) / 'test.bin'
    subprocess_code = textwrap.dedent(f'\n        from memray import Tracker\n        from memray._test import MemoryAllocator\n\n\n        def apple():\n            banana()\n\n\n        def banana():\n            allocator.valloc(1024 * 10)\n            animal.switch()\n            allocator.valloc(1024 * 30)\n\n\n        def ant():\n            allocator.valloc(1024 * 20)\n            fruit.switch()\n            allocator.valloc(1024 * 40)\n            bat()\n            allocator.valloc(1024 * 60)\n\n\n        def bat():\n            allocator.valloc(1024 * 50)\n\n\n        def test():\n            fruit.switch()\n            assert fruit.dead\n            animal.switch()\n            assert animal.dead\n            allocator.valloc(1024 * 70)\n\n\n        allocator = MemoryAllocator()\n        output = "{output}"\n\n        with Tracker(output):\n            import greenlet\n\n            fruit = greenlet.greenlet(apple)\n            animal = greenlet.greenlet(ant)\n            test()\n        ')
    subprocess.run([sys.executable, '-Xdev', '-c', subprocess_code], timeout=5)
    reader = FileReader(output)
    records = list(reader.get_allocation_records())
    vallocs = [record for record in filter_relevant_allocations(records) if record.allocator == AllocatorType.VALLOC]

    def stack(alloc):
        if False:
            for i in range(10):
                print('nop')
        return [frame[0] for frame in alloc.stack_trace()]
    assert stack(vallocs[0]) == ['valloc', 'banana', 'apple']
    assert vallocs[0].size == 10 * 1024
    assert stack(vallocs[1]) == ['valloc', 'ant']
    assert vallocs[1].size == 20 * 1024
    assert stack(vallocs[2]) == ['valloc', 'banana', 'apple']
    assert vallocs[2].size == 30 * 1024
    assert stack(vallocs[3]) == ['valloc', 'ant']
    assert vallocs[3].size == 40 * 1024
    assert stack(vallocs[4]) == ['valloc', 'bat', 'ant']
    assert vallocs[4].size == 50 * 1024
    assert stack(vallocs[5]) == ['valloc', 'ant']
    assert vallocs[5].size == 60 * 1024
    assert stack(vallocs[6]) == ['valloc', 'test', '<module>']
    assert vallocs[6].size == 70 * 1024
    assert vallocs[0].tid != vallocs[1].tid != vallocs[6].tid
    assert vallocs[0].tid == vallocs[2].tid
    assert vallocs[1].tid == vallocs[3].tid == vallocs[4].tid == vallocs[5].tid