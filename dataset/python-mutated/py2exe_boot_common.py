import sys
if sys.frozen == 'windows_exe':

    class Blackhole(object):
        softspace = 0

        def write(self, text):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def flush(self):
            if False:
                return 10
            pass
    sys.stdout = Blackhole()
    sys.stderr = Blackhole()
    del Blackhole
import os
sys.path.append(os.path.join(os.path.dirname(sys.executable), 'site-packages'))
del os
del sys
import linecache

def fake_getline(filename, lineno, module_globals=None):
    if False:
        i = 10
        return i + 15
    return ''
linecache.orig_getline = linecache.getline
linecache.getline = fake_getline
del linecache, fake_getline