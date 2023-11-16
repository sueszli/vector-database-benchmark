import logging
import os
import h2o
import h2o.utils.config
from tests import TemporaryDirectory, pyunit_utils as pu

def test_h2oconfig():
    if False:
        print('Hello World!')
    '\n    Test for parser of the .h2oconfig files.\n\n    This test will create various config files in the tests/results/configtest\n    folder and then parse them with the `H2OConfigReader` class.\n    '
    l = logging.getLogger('h2o')
    l.setLevel(20)
    test_single_config('', {})
    test_single_config('# key = value\n\n', {})
    test_single_config('# key = value\n[init]\n', {})
    test_single_config('\n        [init]\n        check_version = False\n        proxy = http://127.12.34.99.10000\n    ', {'init.check_version': 'False', 'init.proxy': 'http://127.12.34.99.10000'})
    test_single_config('\n        init.check_version = anything!  # rly?\n        init.cookies=A\n        # more comment\n    ', {'init.cookies': 'A', 'init.check_version': 'anything!  # rly?'})
    test_single_config('hbwltqert', {}, n_errors=1)
    test_single_config('\n        init.checkversion = True\n        init.clusterid = 7\n        proxy = None\n    ', {}, n_errors=3)
    test_single_config('\n        [something]\n        init.check_version = True\n    ', {}, 1)
    test_single_config('\n        init.check_version = True\n        init.check_version = False\n        init.check_version = Ambivolent\n    ', {'init.check_version': 'Ambivolent'})

def test_single_config(text, expected, n_errors=0):
    if False:
        return 10
    print()
    with TemporaryDirectory() as target_dir:
        with open(os.path.join(target_dir, '.h2oconfig'), 'wt') as f:
            f.write(text)
        if n_errors:
            print('Expecting %d error%s...' % (n_errors, 's' if n_errors > 1 else ''))
        handler = LogErrorCounter()
        logging.getLogger('h2o').addHandler(handler)
        result = h2o.utils.config.H2OConfigReader(target_dir).read_config()
        assert result == expected, 'Expected config %r but obtained %r' % (expected, result)
        assert handler.errorcount == n_errors, 'Expected %d errors but obtained %d' % (n_errors, handler.errorcount)
        logging.getLogger('h2o').removeHandler(handler)

class LogErrorCounter(logging.Handler):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(self.__class__, self).__init__()
        self.errorcount = 0

    def emit(self, record):
        if False:
            return 10
        if record.levelno >= 40:
            self.errorcount += 1
pu.run_tests([test_h2oconfig])