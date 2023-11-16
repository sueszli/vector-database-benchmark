from builtins import range
import sys
sys.path.insert(1,"../../")
import h2o
from tests import pyunit_utils
import os

from h2o.estimators.gbm import H2OGradientBoostingEstimator

import random

def pub_444_spaces_in_filenames():

    # tempdir = "smalldata/jira/"
    # if was okay to write to smalldata, it's okay to write to the current directory
    # probably don't want to, but can't find what the standard temp directory is supposed to be. no sandbox?
    tempdir = "./"
    # make a few files with spaces in the name
    f1 = open(pyunit_utils.locate(tempdir) + "foo .csv", "w")
    f1.write("response, predictor\n")
    for i in range(10):
        f1.write("1, a\n")
        f1.write("0, b\n")
        f1.write("1, a\n" if random.randint(0,1) else "0, b\n")
    f1.close()

    f2 = open(pyunit_utils.locate(tempdir) + "b a r .csv", "w")
    f2.write("response, predictor\n")
    for i in range(10):
        f2.write("1, a\n")
        f2.write("0, b\n")
        f2.write("1, a\n" if random.randint(0,1) else "0, b\n")
    f2.close()

    f3 = open(pyunit_utils.locate(tempdir) + " ba z.csv", "w")
    for i in range(10):
        f3.write("1, a\n")
        f3.write("0, b\n")
        f3.write("1, a\n" if random.randint(0,1) else "0, b\n")
    f3.close()

    train_data = h2o.upload_file(path=pyunit_utils.locate(tempdir + "foo .csv"))
    train_data.show()
    train_data.describe()
    train_data["response"] = train_data["response"].asfactor()
    gbm = H2OGradientBoostingEstimator(ntrees=1, distribution="bernoulli", min_rows=1)
    gbm.train(x=list(range(1,train_data.ncol)), y="response", training_frame=train_data)
    gbm.show()

    train_data = h2o.upload_file(path=pyunit_utils.locate(tempdir + "b a r .csv"))
    train_data.show()
    train_data.describe()
    train_data["response"] = train_data["response"].asfactor()

    gbm = H2OGradientBoostingEstimator(ntrees=1, distribution="bernoulli", min_rows=1)
    gbm.train(x=1, y="response", training_frame=train_data)

    gbm.show()

    train_data = h2o.upload_file(path=pyunit_utils.locate(tempdir + " ba z.csv"))
    train_data.show()
    train_data.describe()
    train_data[0]=train_data[0].asfactor()
    gbm = H2OGradientBoostingEstimator(ntrees=1, distribution="bernoulli", min_rows=1)
    gbm.train(x=1, y=0, training_frame=train_data)
    gbm.show()

    os.remove(pyunit_utils.locate(tempdir) + "foo .csv")
    os.remove(pyunit_utils.locate(tempdir) + "b a r .csv")
    os.remove(pyunit_utils.locate(tempdir) + " ba z.csv")



if __name__ == "__main__":
    pyunit_utils.standalone_test(pub_444_spaces_in_filenames)
else:
    pub_444_spaces_in_filenames()
