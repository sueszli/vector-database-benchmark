import sys
sys.path.insert(1, '../../')
import h2o
import math
import os
from tests import pyunit_utils

def pubdev_6339():
    if False:
        i = 10
        return i + 15
    cluster = h2o.cluster()
    cloud_size = cluster.cloud_size
    cores = sum((node['num_cpus'] for node in cluster.nodes))
    file_paths = [pyunit_utils.locate('smalldata/arcene/arcene_train.data'), pyunit_utils.locate('smalldata/census_income/adult_data.csv'), pyunit_utils.locate('smalldata/chicago/chicagoAllWeather.csv'), pyunit_utils.locate('smalldata/gbm_test/alphabet_cattest.csv'), pyunit_utils.locate('smalldata/wa_cannabis/raw/Dashboard_Usable_Sales_w_Weight_Daily.csv')]
    for file_path in file_paths:
        data_raw = h2o.import_file(path=file_path, parse=False)
        setup = h2o.parse_setup(data_raw)
        num_cols = setup['number_columns']
        chunk_size = calculate_chunk_size(file_path, num_cols, cores, cloud_size)
        result_size = setup['chunk_size']
        assert chunk_size == result_size, 'Calculated chunk size is incorrect!'
        print('chunk size for file', file_path, 'is:', chunk_size)
    data_raw = h2o.import_file(path=file_paths[1], parse=False)
    setup = h2o.parse_setup(data_raw)

def calculate_chunk_size(file_path, num_cols, cores, cloud_size):
    if False:
        while True:
            i = 10
    '\n        Return size of a chunk calculated for optimal data handling in h2o java backend.\n    \n        :param file_path:  path to dataset\n        :param num_cols:  number or columns in dataset\n        :param cores:  number of CPUs on machine where the model was trained\n        :param cloud_size:  number of nodes on machine where the model was trained\n        :return:  a chunk size \n    '
    max_line_length = 0
    total_size = 0
    with open(file_path, 'r') as input_file:
        for line in input_file:
            size = len(line)
            total_size = total_size + size
            if size > max_line_length:
                max_line_length = size
    default_log2_chunk_size = 20 + 2
    default_chunk_size = 1 << default_log2_chunk_size
    local_parse_size = int(total_size / cloud_size)
    min_number_rows = 10
    per_node_chunk_count_limit = 1 << 21
    min_parse_chunk_size = 1 << 12
    max_parse_chunk_size = (1 << 28) - 1
    chunk_size = int(max(local_parse_size / (4 * cores) + 1, min_parse_chunk_size))
    if chunk_size > 1024 * 1024:
        chunk_size = (chunk_size & 4294966784) + 512
    if total_size <= 1 << 16:
        chunk_size = max(default_chunk_size, int(min_number_rows * max_line_length))
    elif chunk_size < default_chunk_size and local_parse_size / chunk_size * num_cols < per_node_chunk_count_limit:
        chunk_size = max(int(chunk_size), int(min_number_rows * max_line_length))
    else:
        chunk_count = cores * 4 * num_cols
        if chunk_count > per_node_chunk_count_limit:
            ratio = 1 << max(2, int(math.log(int(chunk_count / per_node_chunk_count_limit), 2)))
            chunk_size = chunk_size * ratio
        chunk_size = min(max_parse_chunk_size, chunk_size)
        if chunk_size <= min_number_rows * max_line_length:
            chunk_size = int(max(default_chunk_size, min(max_parse_chunk_size, min_number_rows * max_line_length)))
    return int(chunk_size)
if __name__ == '__main__':
    pyunit_utils.standalone_test(pubdev_6339)
else:
    pubdev_6339()