import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import time
import random

def test_arrange_OOM():
    if False:
        return 10
    '\n    PUBDEV-5990 customer reported that h2o.arrange (sorting) takes way more memory than normal for sparse\n    datasets of 1G.\n\n    Thanks to Lauren DiPerna for finding the dataset to repo the problem.\n    '
    df = h2o.import_file(pyunit_utils.locate('bigdata/laptop/jira/sort_OOM.csv'))
    t1 = time.time()
    newFrame = df.sort('sort_col')
    print(newFrame[0, 0])
    elapsed_time = time.time() - t1
    print('time taken to perform sort is {0}'.format(elapsed_time))
    answerFrame = h2o.import_file(pyunit_utils.locate('bigdata/laptop/jira/sort_OOM_answer.csv'))
    pyunit_utils.compare_frames_local(answerFrame['sort_col'], newFrame['sort_col'])
    allColumns = list(range(0, df.ncols))
    random.shuffle(allColumns)
    pyunit_utils.compare_frames_local(answerFrame[allColumns[0:5]], newFrame[allColumns[0:5]])
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_arrange_OOM)
else:
    test_arrange_OOM()