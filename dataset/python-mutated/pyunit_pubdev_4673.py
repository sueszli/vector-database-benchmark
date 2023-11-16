import h2o
from tests import pyunit_utils

def test_4673():
    if False:
        for i in range(10):
            print('nop')
    fr = h2o.import_file(pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    print(fr.mean())
    fr.apply(lambda x: x['class'] + x[0], axis=1).show()
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_4673)
else:
    test_4673()