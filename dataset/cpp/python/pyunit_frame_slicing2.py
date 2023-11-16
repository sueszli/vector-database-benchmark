import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils




def expr_slicing():

    iris = h2o.import_file(path=pyunit_utils.locate("smalldata/iris/iris_wheader.csv"))
    iris.show()

    ###################################################################

    # H2OFrame[int] (column slice)
    res = 2 - iris
    res2 = res[0]
    assert abs(res2[3,:] - -2.6) < 1e-10 and abs(res2[17,:] - -3.1) < 1e-10 and abs(res2[24,:] - -2.8) < 1e-10, \
        "incorrect values"

    # H2OFrame[int,int]
    assert abs(res[13, 3] - 1.9) < 1e-10, "incorrect values"

    # H2OFrame[int, slice]
    res4 = res[12, 0:4]
    assert abs(res4[0,0] - -2.8) < 1e-10 and abs(res4[0,1] - -1.0) < 1e-10 and abs(res4[0,2] - 0.6) < 1e-10 and \
        abs(res4[0,3] - 1.9) < 1e-10, "incorrect values"

    # H2OFrame[slice, int]
    res5 = res[5:9, 1]
    assert abs(res5[0,:] - -1.9) < 1e-10 and abs(res5[1,:] - -1.4) < 1e-10 and abs(res5[2,:] - -1.4) < 1e-10 and \
           abs(res5[3,:] - -0.9) < 1e-10, "incorrect values"

    # H2OFrame[slice, slice]
    res = iris * 2
    res6 = res[5:9, 0:4]
    assert abs(res6[0,0] - 10.8) < 1e-10 and abs(res6[1,1] - 6.8) < 1e-10 and abs(res6[2,2] - 3.0) < 1e-10 and \
           abs(res6[3,3] - 0.4) < 1e-10, "incorrect values"



if __name__ == "__main__":
    pyunit_utils.standalone_test(expr_slicing)
else:
    expr_slicing()
