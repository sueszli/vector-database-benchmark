from builtins import zip
import sys
sys.path.insert(1,"../../")
import h2o
from tests import pyunit_utils
# Check to make sure the small and large citibike demos have not diverged
import os

def consistency_check():

    try:
        small = pyunit_utils.locate("h2o-py/demos/citi_bike_small.ipynb")
    except ValueError:
        small = pyunit_utils.locate("h2o-py/demos/citi_bike_small_NOPASS.ipynb")

    try:
        large = pyunit_utils.locate("h2o-py/demos/citi_bike_large.ipynb")
    except ValueError:
        large = pyunit_utils.locate("h2o-py/demos/citi_bike_large_NOPASS.ipynb")

    results_dir = pyunit_utils.locate("results")
    s = os.path.join(results_dir, os.path.basename(small).split('.')[0]+".py")
    l = os.path.join(results_dir, os.path.basename(large).split('.')[0]+".py")

    from tests import pydemo_utils
    pydemo_utils.ipy_notebook_exec(small, save_and_norun = s)
    pydemo_utils.ipy_notebook_exec(large, save_and_norun = l)

    small_list = list(open(s, 'r'))
    large_list = list(open(l, 'r'))

    for s, l in zip(small_list, large_list):
        if s != l:
            assert s == "data = h2o.import_file(path=small_test)\n" and \
                   l != "data = h2o.import_file(path=large_test)\n", \
                "This difference is not allowed between the small and large citibike demos.\nCitibike small: {0}" \
                "Citibike large: {1}".format(s,l)



if __name__ == "__main__":
    pyunit_utils.standalone_test(consistency_check)
else:
    consistency_check()
