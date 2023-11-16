import sys
sys.path.insert(1, '../..')
from tests import pyunit_utils
import h2o

def read_1gb_cloud():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test h2o cluster read file.  Should run faster than what is observed under\n    https://github.com/h2oai/h2o-3/issues/15163\n    Right now this test is not run through Jenkins. Need to setup a cloud\n    testing infrastructure which is a longer term project.\n    You can take a look at markc_multimachine on jenkins for the current setup\n    which is based on ec2\n    '
    df = h2o.import_file('http://s3.amazonaws.com/h2o-datasets/allstate/train_set.zip')
    response = 'Cat1'
    predictors = ['Cat2', 'Cat3', 'Cat4', 'Cat5']
    df['Cat1'] = df['Cat1'].asfactor()
    df['Cat1'].summary()
    rnd = df['Cat1'].runif(seed=1234)
    train = df[rnd <= 0.8]
    test = df[rnd > 0.8]
if __name__ == '__main__':
    pyunit_utils.standalone_test(read_1gb_cloud)
else:
    read_1gb_cloud()