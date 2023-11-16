import sys
import os
sys.path.insert(1, os.path.join("..", "..", "..", "h2o-py"))
from tests import pyunit_utils
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator

# Purpose:  This tests k-means on a large dataset.

def hdfs_kmeans():
    hdfs_name_node = pyunit_utils.hadoop_namenode()
    hdfs_iris_file = "/datasets/runit/iris_wheader.csv"
    hdfs_covtype_file = "/datasets/runit/covtype.data"

    print("Import iris_wheader.csv from HDFS")
    url = "hdfs://{0}{1}".format(hdfs_name_node, hdfs_iris_file)
    iris_h2o = h2o.import_file(url)
    n = iris_h2o.nrow
    print("rows: {0}".format(n))
    assert n == 150, "Wrong number of rows. Got {0}. Should have got {1}".format(n, 150)

    print("Running KMeans on iris")
    iris_km = H2OKMeansEstimator(k=3, training_frame=iris_h2o[0:4], max_iterations=10)
    iris_km.train()
    print(iris_km)

    print("Importing covtype.data from HDFS")
    url = "hdfs://{0}{1}".format(hdfs_name_node, hdfs_covtype_file)
    covtype_h2o = h2o.import_file(url)
    n = covtype_h2o.nrow
    print("rows: {0}".format(n))
    assert n == 581012, "Wrong number of rows. Got {0}. Should have got {1}".format(n, 581012)

    print("Running KMeans on covtype")
    covtype_km = H2OKMeansEstimator(training_frame=covtype_h2o[0:55], k=8, max_iterations=10)
    covtype_km.train()
    print(covtype_km)


if __name__ == "__main__":
    pyunit_utils.standalone_test(hdfs_kmeans)
else:
    hdfs_kmeans()
