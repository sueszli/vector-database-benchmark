import h2o
from tests import pyunit_utils
import sys

def pubdev_6304():
    if False:
        for i in range(10):
            print('nop')
    fractions = dict()
    fractions['real_fraction'] = 0
    fractions['categorical_fraction'] = 1
    fractions['integer_fraction'] = 0
    fractions['time_fraction'] = 0
    fractions['string_fraction'] = 0
    fractions['binary_fraction'] = 0
    try:
        traindata = h2o.create_frame(rows=100, cols=2, missing_fraction=0, has_response=False, factors=9999999, seed=12345, **fractions)
    except Exception as ex:
        sys.exit(1)
    try:
        traindata = h2o.create_frame(rows=100, cols=2, missing_fraction=0, has_response=False, factors=19999999, seed=12345, **fractions)
        sys.exit(1)
    except Exception as ex:
        print(ex)
        if 'Number of factors must be <= 10,000,000' in ex.args[0].dev_msg:
            sys.exit(0)
        else:
            sys.exit(1)
if __name__ == '__main__':
    pyunit_utils.standalone_test(pubdev_6304)
else:
    pubdev_6304()