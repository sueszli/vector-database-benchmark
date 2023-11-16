import sys
sys.path.insert(1, '../../')
import h2o
from h2o.exceptions import H2OTypeError
from tests import pyunit_utils
import urllib.request, urllib.parse, urllib.error

def separator():
    if False:
        while True:
            i = 10
    path = 'smalldata/jira/hexdev_29.csv'
    fhex = h2o.import_file(pyunit_utils.locate(path), sep=',')
    fhex.summary()
    fhex_col_summary = h2o.api('GET /3/Frames/%s/summary' % urllib.parse.quote(fhex.frame_id))['frames'][0]['columns']
    fhex_missing_count = sum((e['missing_count'] for e in fhex_col_summary))
    assert fhex_missing_count == 0
    fhex_wrong_separator = h2o.import_file(pyunit_utils.locate(path), sep=';')
    fhex_wrong_separator.summary()
    assert fhex_wrong_separator.ncol == 1
    assert fhex_wrong_separator.nrow == 6
    try:
        h2o.import_file(pyunit_utils.locate(path), sep='--')
    except H2OTypeError:
        pass
    else:
        assert False
if __name__ == '__main__':
    pyunit_utils.standalone_test(separator)
else:
    separator()