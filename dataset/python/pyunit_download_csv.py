import sys
sys.path.insert(1,"../../")
import h2o
from tests import pyunit_utils
import os

import random

def download_csv():
    
    

    iris1 = h2o.import_file(path=pyunit_utils.locate("smalldata/iris/iris.csv"))

    h2o.download_csv(iris1,"iris_delete.csv")

    iris2 = h2o.import_file(path=pyunit_utils.locate("iris_delete.csv"))
    os.remove("iris_delete.csv")

    rand_row = random.randint(0,iris1.nrow-1)
    rand_col = random.randint(0,3)
    assert abs(iris1[rand_row, rand_col] - iris2[rand_row, rand_col]) < 1e-10, "Expected elements from the datasets to " \
                                                                                "be the same, but got {0} and {1}" \
                                                                                "".format(iris1[rand_row, rand_col],
                                                                                          iris2[rand_row, rand_col])


if __name__ == "__main__":
    pyunit_utils.standalone_test(download_csv)
else:
    download_csv()
