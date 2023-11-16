from test import support
from test.support import import_helper
import unittest
nis = import_helper.import_module('nis')

class NisTests(unittest.TestCase):

    def test_maps(self):
        if False:
            print('Hello World!')
        try:
            maps = nis.maps()
        except nis.error as msg:
            self.skipTest(str(msg))
        try:
            maps.remove('passwd.adjunct.byname')
        except ValueError:
            pass
        done = 0
        for nismap in maps:
            mapping = nis.cat(nismap)
            for (k, v) in mapping.items():
                if not k:
                    continue
                if nis.match(k, nismap) != v:
                    self.fail("NIS match failed for key `%s' in map `%s'" % (k, nismap))
                else:
                    done = 1
                    break
            if done:
                break
if __name__ == '__main__':
    unittest.main()