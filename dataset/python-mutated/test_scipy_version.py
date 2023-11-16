import re
import scipy
from numpy.testing import assert_

def test_valid_scipy_version():
    if False:
        i = 10
        return i + 15
    version_pattern = '^[0-9]+\\.[0-9]+\\.[0-9]+(|a[0-9]|b[0-9]|rc[0-9])'
    dev_suffix = '(\\.dev0\\+.+([0-9a-f]{7}|Unknown))'
    if scipy.version.release:
        res = re.match(version_pattern, scipy.__version__)
    else:
        res = re.match(version_pattern + dev_suffix, scipy.__version__)
    assert_(res is not None, scipy.__version__)