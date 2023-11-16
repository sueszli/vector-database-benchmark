import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import urllib.request, urllib.parse, urllib.error

def na_strings():
    if False:
        for i in range(10):
            print('nop')
    path = 'smalldata/jira/hexdev_29.csv'
    fhex = h2o.import_file(pyunit_utils.locate(path))
    fhex.summary()
    fhex_col_summary = h2o.api('GET /3/Frames/%s/summary' % urllib.parse.quote(fhex.frame_id))['frames'][0]['columns']
    fhex_missing_count = sum([e['missing_count'] for e in fhex_col_summary])
    assert fhex_missing_count == 0
    fhex_na_strings = h2o.import_file(pyunit_utils.locate(path), na_strings=[[], ['fish', 'xyz'], []])
    fhex_na_strings.summary()
    fhex__na_strings_col_summary = h2o.api('GET /3/Frames/%s/summary' % urllib.parse.quote(fhex_na_strings.frame_id))['frames'][0]['columns']
    fhex_na_strings_missing_count = sum([e['missing_count'] for e in fhex__na_strings_col_summary])
    assert fhex_na_strings_missing_count == 2
    fhex_na_strings = h2o.import_file(pyunit_utils.locate(path), na_strings=['fish', 'xyz'])
    fhex_na_strings.summary()
    fhex__na_strings_col_summary = h2o.api('GET /3/Frames/%s/summary' % urllib.parse.quote(fhex_na_strings.frame_id))['frames'][0]['columns']
    fhex_na_strings_missing_count = sum([e['missing_count'] for e in fhex__na_strings_col_summary])
    assert fhex_na_strings_missing_count == 2
    fhex_na_strings = h2o.import_file(pyunit_utils.locate(path), na_strings={'h2': 'fish'})
    fhex_na_strings.summary()
    fhex__na_strings_col_summary = h2o.api('GET /3/Frames/%s/summary' % urllib.parse.quote(fhex_na_strings.frame_id))['frames'][0]['columns']
    fhex_na_strings_missing_count = sum([e['missing_count'] for e in fhex__na_strings_col_summary])
    assert fhex_na_strings_missing_count == 2
    fhex_na_strings = h2o.import_file(pyunit_utils.locate(path), na_strings={'h1': 'fish'})
    fhex_na_strings.summary()
    fhex__na_strings_col_summary = h2o.api('GET /3/Frames/%s/summary' % urllib.parse.quote(fhex_na_strings.frame_id))['frames'][0]['columns']
    fhex_na_strings_missing_count = sum([e['missing_count'] for e in fhex__na_strings_col_summary])
    assert fhex_na_strings_missing_count == 0
    fhex_na_strings = h2o.import_file(pyunit_utils.locate(path), na_strings={'h2': ['fish', 'xyz']})
    fhex_na_strings.summary()
    fhex__na_strings_col_summary = h2o.api('GET /3/Frames/%s/summary' % urllib.parse.quote(fhex_na_strings.frame_id))['frames'][0]['columns']
    fhex_na_strings_missing_count = sum([e['missing_count'] for e in fhex__na_strings_col_summary])
    assert fhex_na_strings_missing_count == 2
if __name__ == '__main__':
    pyunit_utils.standalone_test(na_strings)
else:
    na_strings()