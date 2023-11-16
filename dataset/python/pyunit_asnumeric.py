import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils




def asnumeric():
    

    h2oframe =  h2o.import_file(path=pyunit_utils.locate("smalldata/junit/cars.csv"))
    rows = h2oframe.nrow

    # H2OFrame case
    h2oframe = h2oframe.asnumeric()
    h2oframe['cylinders'] = h2oframe['cylinders'] - h2oframe['cylinders']
    h2oframe = h2oframe[h2oframe['cylinders'] == 0]
    assert h2oframe.nrow == rows, "expected the same number of rows as before {0}, but got {1}".format(rows, h2oframe.nrow)

    h2oframe =  h2o.import_file(path=pyunit_utils.locate("smalldata/junit/cars.csv"))

    # H2OVec case
    h2oframe['cylinders'] = h2oframe['cylinders'].asnumeric()
    h2oframe['cylinders'] = h2oframe['cylinders'] - h2oframe['cylinders']
    h2oframe = h2oframe[h2oframe['cylinders'] == 0]
    assert h2oframe.nrow == rows, "expected the same number of rows as before {0}, but got {1}".format(rows, h2oframe.nrow)



if __name__ == "__main__":
    pyunit_utils.standalone_test(asnumeric)
else:
    asnumeric()
