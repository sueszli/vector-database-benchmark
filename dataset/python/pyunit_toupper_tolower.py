import sys
sys.path.insert(1,"../../")
import h2o
from tests import pyunit_utils




def toupper_tolower_check():
    # Connect to a pre-existing cluster
    

    frame = h2o.import_file(path=pyunit_utils.locate("smalldata/iris/iris.csv"), col_types=["numeric","numeric","numeric","numeric","string"])

    # single column (frame)
    frame["C5"] = frame["C5"].toupper()
    assert frame[0,4] == "IRIS-SETOSA", "Expected 'IRIS-SETOSA', but got {0}".format(frame[0,4])

    # single column (vec)
    vec = frame["C5"]
    vec = vec.toupper()
    assert vec[2,0] == "IRIS-SETOSA", "Expected 'IRIS-SETOSA', but got {0}".format(vec[2,0])

    vec = vec.tolower()
    assert vec[3,0] == "iris-setosa", "Expected 'iris-setosa', but got {0}".format(vec[3,0])



if __name__ == "__main__":
    pyunit_utils.standalone_test(toupper_tolower_check)
else:
    toupper_tolower_check()
