"""Define data shapes."""
import json
import os
from .compatibility import ASV_DATASET_SIZE, ASV_USE_STORAGE_FORMAT
RAND_LOW = 0
RAND_HIGH = 1000000000 if ASV_USE_STORAGE_FORMAT == 'hdk' and ASV_DATASET_SIZE == 'Big' else 100
BINARY_OP_DATA_SIZE = {'big': [[[5000, 5000], [5000, 5000]], [[500000, 20], [1000000, 10]]], 'small': [[[250, 250], [250, 250]], [[10000, 20], [25000, 10]]]}
UNARY_OP_DATA_SIZE = {'big': [[5000, 5000], [1000000, 10]], 'small': [[250, 250], [10000, 10]]}
SERIES_DATA_SIZE = {'big': [[100000, 1]], 'small': [[10000, 1]]}
BINARY_OP_SERIES_DATA_SIZE = {'big': [[[500000, 1], [1000000, 1]], [[500000, 1], [500000, 1]]], 'small': [[[5000, 1], [10000, 1]]]}
HDK_BINARY_OP_DATA_SIZE = {'big': [[[500000, 20], [1000000, 10]]], 'small': [[[10000, 20], [25000, 10]]]}
HDK_UNARY_OP_DATA_SIZE = {'big': [[1000000, 10]], 'small': [[10000, 10]]}
HDK_SERIES_DATA_SIZE = {'big': [[10000000, 1]], 'small': [[100000, 1]]}
DEFAULT_GROUPBY_NGROUPS = {'big': [100, 'huge_amount_groups'], 'small': [5]}
GROUPBY_NGROUPS = DEFAULT_GROUPBY_NGROUPS[ASV_DATASET_SIZE]
_DEFAULT_CONFIG_T = [(UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE], ['TimeGroupByMultiColumn', 'TimeGroupByDefaultAggregations', 'TimeGroupByDictionaryAggregation', 'TimeSetItem', 'TimeInsert', 'TimeArithmetic', 'TimeSortValues', 'TimeDrop', 'TimeHead', 'TimeTail', 'TimeExplode', 'TimeFillna', 'TimeFillnaDataFrame', 'TimeValueCountsFrame', 'TimeValueCountsSeries', 'TimeIndexing', 'TimeMultiIndexing', 'TimeResetIndex', 'TimeAstype', 'TimeDescribe', 'TimeProperties', 'TimeReindex', 'TimeReindexMethod', 'TimeFillnaMethodDataframe', 'TimeDropDuplicatesDataframe', 'TimeStack', 'TimeUnstack', 'TimeRepr', 'TimeMaskBool', 'TimeIsnull', 'TimeDropna', 'TimeEquals', 'TimeReadCsvSkiprows', 'TimeReadCsvTrueFalseValues', 'TimeReadCsvNamesDtype', 'TimeReadParquet', 'TimeFromPandas', 'TimeToPandas', 'TimeToNumPy']), (BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE], ['TimeJoin', 'TimeMerge', 'TimeMergeDefault', 'TimeConcat', 'TimeAppend', 'TimeBinaryOp', 'TimeLevelAlign']), (SERIES_DATA_SIZE[ASV_DATASET_SIZE], ['TimeFillnaSeries', 'TimeGroups', 'TimeIndexingNumericSeries', 'TimeFillnaMethodSeries', 'TimeDatetimeAccessor', 'TimeSetCategories', 'TimeRemoveCategories', 'TimeDropDuplicatesSeries']), (BINARY_OP_SERIES_DATA_SIZE[ASV_DATASET_SIZE], ['TimeBinaryOpSeries'])]
_DEFAULT_HDK_CONFIG_T = [(HDK_UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE], ['hdk.TimeJoin', 'hdk.TimeBinaryOpDataFrame', 'hdk.TimeArithmetic', 'hdk.TimeSortValues', 'hdk.TimeDrop', 'hdk.TimeHead', 'hdk.TimeFillna', 'hdk.TimeIndexing', 'hdk.TimeResetIndex', 'hdk.TimeAstype', 'hdk.TimeDescribe', 'hdk.TimeProperties', 'hdk.TimeGroupByDefaultAggregations', 'hdk.TimeGroupByMultiColumn', 'hdk.TimeValueCountsDataFrame', 'hdk.TimeReadCsvNames']), (HDK_BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE], ['hdk.TimeMerge', 'hdk.TimeAppend']), (HDK_SERIES_DATA_SIZE[ASV_DATASET_SIZE], ['hdk.TimeBinaryOpSeries', 'hdk.TimeValueCountsSeries'])]
DEFAULT_CONFIG = {}
DEFAULT_CONFIG['MergeCategoricals'] = [[10000, 2]] if ASV_DATASET_SIZE == 'big' else [[1000, 2]]
DEFAULT_CONFIG['TimeJoinStringIndex'] = [[100000, 64]] if ASV_DATASET_SIZE == 'big' else [[1000, 4]]
DEFAULT_CONFIG['TimeReplace'] = [[10000, 2]] if ASV_DATASET_SIZE == 'big' else [[1000, 2]]
for config in (_DEFAULT_CONFIG_T, _DEFAULT_HDK_CONFIG_T):
    for (_shape, _names) in config:
        DEFAULT_CONFIG.update({_name: _shape for _name in _names})
if ASV_DATASET_SIZE == 'big':
    DEFAULT_CONFIG['TimeMergeDefault'] = [[[1000, 1000], [1000, 1000]], [[500000, 20], [1000000, 10]]]
    DEFAULT_CONFIG['TimeLevelAlign'] = [[[2500, 2500], [2500, 2500]], [[250000, 20], [500000, 10]]]
    DEFAULT_CONFIG['TimeStack'] = [[1500, 1500], [100000, 10]]
    DEFAULT_CONFIG['TimeUnstack'] = DEFAULT_CONFIG['TimeStack']
CONFIG_FROM_FILE = None

def get_benchmark_shapes(bench_id: str):
    if False:
        i = 10
        return i + 15
    '\n    Get custom benchmark shapes from a json file stored in MODIN_ASV_DATASIZE_CONFIG.\n\n    If `bench_id` benchmark is not found in the file, then the default value will\n    be used.\n\n    Parameters\n    ----------\n    bench_id : str\n        Unique benchmark identifier that is used to get shapes.\n\n    Returns\n    -------\n    list\n        Benchmark shapes.\n    '
    global CONFIG_FROM_FILE
    if not CONFIG_FROM_FILE:
        try:
            from modin.config import AsvDataSizeConfig
            filename = AsvDataSizeConfig.get()
        except ImportError:
            filename = os.environ.get('MODIN_ASV_DATASIZE_CONFIG', None)
        if filename:
            with open(filename) as _f:
                CONFIG_FROM_FILE = json.load(_f)
    if CONFIG_FROM_FILE and bench_id in CONFIG_FROM_FILE:
        return CONFIG_FROM_FILE[bench_id]
    return DEFAULT_CONFIG[bench_id]