import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils

def anyfactor():
    if False:
        print('Hello World!')
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris.csv'))
    assert iris.anyfactor(), 'Expected true, but got false. Column 5 is a factor.'
    assert not iris[:, :4].anyfactor(), 'Expected false, but got true. Columns 1-4 are numeric.'
    assert iris[4].anyfactor(), 'Expected true, but got false. Column 5 is a factor.'
    assert not iris[0].anyfactor(), 'Expected false, but got true. Columns 1 is numeric.'
if __name__ == '__main__':
    pyunit_utils.standalone_test(anyfactor)
else:
    anyfactor()