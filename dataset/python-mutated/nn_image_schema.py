import sys
from bigdl.dllib.utils.file_utils import callZooFunc
if sys.version >= '3':
    long = int
    unicode = str

def with_origin_column(dataset, imageColumn='image', originColumn='origin', bigdl_type='float'):
    if False:
        print('Hello World!')
    return callZooFunc(bigdl_type, 'withOriginColumn', dataset, imageColumn, originColumn)