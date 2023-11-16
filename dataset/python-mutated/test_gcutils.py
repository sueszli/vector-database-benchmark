import sys
import time
if '__pypy__' not in sys.builtin_module_names:
    from boltons.gcutils import get_all, toggle_gc_postcollect

    def test_get_all():
        if False:
            return 10

        class TestType(object):
            pass
        tt = TestType()
        assert len(get_all(TestType)) == 1
        assert len(get_all(bool)) == 0
        return

    def test_toggle_gc_postcollect():
        if False:
            for i in range(10):
                print('nop')
        COUNT = int(1000000.0)
        start = time.time()
        with toggle_gc_postcollect:
            x = [{} for x in range(COUNT)]
        no_gc_time = time.time() - start
        start = time.time()
        x = [{} for x in range(COUNT)]
        with_gc_time = time.time() - start
        time_diff = no_gc_time < with_gc_time