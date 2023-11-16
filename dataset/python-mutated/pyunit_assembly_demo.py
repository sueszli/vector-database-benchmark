import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.assembly import H2OAssembly
from h2o.transforms.preprocessing import *
from h2o import H2OFrame

def assembly_demo():
    if False:
        return 10
    fr = h2o.import_file(pyunit_utils.locate('smalldata/iris/iris_wheader.csv'), col_types=['numeric', 'numeric', 'numeric', 'numeric', 'string'])
    assembly = H2OAssembly(steps=[('col_select', H2OColSelect(['sepal_len', 'petal_len', 'class'])), ('cos_sep_len', H2OColOp(op=H2OFrame.cos, col='sepal_len', inplace=True)), ('str_cnt_species', H2OColOp(op=H2OFrame.countmatches, col='class', inplace=False, pattern='s'))])
    result = assembly.fit(fr)
    result.show()
    assembly.to_pojo('MungingPojoDemo')
if __name__ == '__main__':
    pyunit_utils.standalone_test(assembly_demo)
else:
    assembly_demo()