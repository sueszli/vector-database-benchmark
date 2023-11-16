import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def import_folder():
    if False:
        for i in range(10):
            print('nop')
    cars = h2o.import_file(path=pyunit_utils.locate('smalldata/synthetic_perfect_separation'))
    print(cars.head())
    cars = h2o.import_file(path=pyunit_utils.locate('smalldata/synthetic_perfect_separation/'))
    print(cars.head())
if __name__ == '__main__':
    pyunit_utils.standalone_test(import_folder)
else:
    import_folder()