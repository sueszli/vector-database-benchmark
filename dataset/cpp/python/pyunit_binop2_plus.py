from builtins import zip
from builtins import range
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils




def binop_plus():
    
    

    iris = h2o.import_file(path=pyunit_utils.locate("smalldata/iris/iris_wheader_65_rows.csv"))
    rows, cols = iris.dim
    iris.show()

    ###################################################################

    # LHS: scaler, RHS: H2OFrame
    res = 2 + iris
    res_rows, res_cols = res.dim
    assert res_rows == rows and res_cols == cols, "dimension mismatch"
    for x, y in zip([res[c].sum() for c in range(cols-1)], [469.9, 342.6, 266.9, 162.2]):
        assert abs(x - y) < 1e-1,  "expected same values"

    # LHS: scaler, RHS: scaler
    res = 2 + iris[0]
    res2 = 1.1 + res[21,:]
    assert abs(res2 - 8.2) < 1e-1, "expected same values"

    ###################################################################

    # LHS: scaler, RHS: H2OFrame
    # res = 1.2 + iris[2]
    # res2 = res[21,:] + iris
    # res2.show()


    # LHS: scaler, RHS: H2OVec
    # res = 1.2 + iris[2]
    # res2 = res[21,:] + iris[1]
    # res2.show()

    # LHS: scaler, RHS: scaler
    res = 1.1 + iris[2]
    res2 = res[21,:] + res[10,:]
    assert abs(res2 - 5.2) < 1e-1, "expected same values"

    # LHS: scaler, RHS: scaler
    res = 2 + iris[0]
    res2 = res[21,:] + 3
    assert abs(res2 - 10.1) < 1e-1, "expected same values"

    ###################################################################

    # LHS: H2OVec, RHS: H2OFrame
    #try:
    #    res = iris[2] + iris
    #    res.show()
    #    assert False, "expected error. objects with different dimensions not supported."
    #except EnvironmentError:
    #    pass

    # LHS: H2OVec, RHS: scaler
    # res = 1.2 + iris[2]
    # res2 = iris[1] + res[21,:]
    # res2.show()

    ###################################################################

    # LHS: H2OFrame, RHS: H2OFrame
    res = iris + iris
    res_rows, res_cols = res.dim
    assert res_rows == rows and res_cols == cols, "dimension mismatch"

    res = iris[0:2] + iris[1:3]
    res_rows, res_cols = res.dim
    assert res_rows == rows and res_cols == 2, "dimension mismatch"

    #try:
    #    res = iris + iris[0:3]
    #    res.show()
    #    assert False, "expected error. frames are different dimensions."
    #except EnvironmentError:
    #    pass

    # LHS: H2OFrame, RHS: H2OVec
    #try:
    #    res = iris + iris[0]
    #    res.show()
    #    assert False, "expected error. objects of different dimensions not supported."
    #except EnvironmentError:
    #    pass

    # LHS: H2OFrame, RHS: scaler
    # res = 1.2 + iris[2]
    # res2 = iris + res[21,:]
    # res2.show()

    # LHS: H2OFrame, RHS: scaler
    res = iris + 2
    res_rows, res_cols = res.dim
    assert res_rows == rows and res_cols == cols, "dimension mismatch"
    for x, y in zip([res[c].sum() for c in range(cols-1)], [469.9, 342.6, 266.9, 162.2]):
        assert abs(x - y) < 1e-1,  "expected same values"

    ###################################################################



if __name__ == "__main__":
    pyunit_utils.standalone_test(binop_plus)
else:
    binop_plus()
