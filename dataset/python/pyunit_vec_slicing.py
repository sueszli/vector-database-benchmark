import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils




def vec_slicing():
    
    

    iris = h2o.import_file(path=pyunit_utils.locate("smalldata/iris/iris_wheader.csv"))
    iris.show()

    ###################################################################

    # H2OVec[int]
    res = 2 - iris
    res2 = res[0]
    assert abs(res2[3,0] - -2.6) < 1e-10 and abs(res2[17,0] - -3.1) < 1e-10 and abs(res2[24,0] - -2.8) < 1e-10, "incorrect values"

    # H2OVec[slice]
    res = iris[12:25,1]
    assert abs(res[0,0] - 3.0) < 1e-10 and abs(res[1,0] - 3.0) < 1e-10 and abs(res[5,0] - 3.5) < 1e-10, "incorrect values"



if __name__ == "__main__":
    pyunit_utils.standalone_test(vec_slicing)
else:
    vec_slicing()
