import os
import sys
import unittest
from ctypes.macholib.dyld import dyld_find

def find_lib(name):
    if False:
        while True:
            i = 10
    possible = ['lib' + name + '.dylib', name + '.dylib', name + '.framework/' + name]
    for dylib in possible:
        try:
            return os.path.realpath(dyld_find(dylib))
        except ValueError:
            pass
    raise ValueError('%s not found' % (name,))

class MachOTest(unittest.TestCase):

    @unittest.skipUnless(sys.platform == 'darwin', 'OSX-specific test')
    def test_find(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIn(find_lib('pthread'), ('/usr/lib/libSystem.B.dylib', '/usr/lib/libpthread.dylib'))
        result = find_lib('z')
        self.assertRegex(result, '.*/lib/libz.*\\.dylib')
        self.assertIn(find_lib('IOKit'), ('/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit', '/System/Library/Frameworks/IOKit.framework/IOKit'))
if __name__ == '__main__':
    unittest.main()