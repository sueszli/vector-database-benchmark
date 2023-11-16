"""
Fodder for module finalization tests in test_module.
"""
import shutil
import test.final_a
x = 'b'

class C:

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        print('x =', x)
        print('final_a.x =', test.final_a.x)
        print('shutil.rmtree =', getattr(shutil.rmtree, '__name__', None))
        print('len =', getattr(len, '__name__', None))
c = C()
_underscored = C()