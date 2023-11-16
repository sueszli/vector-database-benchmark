import argparse
import yaml
from api_gen import ForwardAPI
from dist_api_gen import DistForwardAPI
from sparse_api_gen import SparseAPI

def header_include():
    if False:
        return 10
    return '\n#include <tuple>\n\n#include "paddle/phi/api/include/tensor.h"\n#include "paddle/phi/common/scalar.h"\n#include "paddle/phi/common/int_array.h"\n#include "paddle/utils/optional.h"\n'

def source_include(header_file_path):
    if False:
        return 10
    return f'#include "{header_file_path}"\n\n#include <memory>\n\n#include "glog/logging.h"\n#include "paddle/utils/flags.h"\n\n#include "paddle/phi/api/lib/api_custom_impl.h"\n#include "paddle/phi/api/lib/api_gen_utils.h"\n#include "paddle/phi/api/lib/data_transform.h"\n#include "paddle/phi/api/lib/kernel_dispatch.h"\n#include "paddle/phi/core/kernel_registry.h"\n#include "paddle/phi/infermeta/binary.h"\n#include "paddle/phi/infermeta/multiary.h"\n#include "paddle/phi/infermeta/nullary.h"\n#include "paddle/phi/infermeta/unary.h"\n#include "paddle/phi/infermeta/ternary.h"\n\n#include "paddle/phi/infermeta/sparse/unary.h"\n#include "paddle/phi/infermeta/sparse/binary.h"\n#include "paddle/phi/infermeta/sparse/multiary.h"\n\n#include "paddle/phi/api/profiler/event_tracing.h"\n#include "paddle/phi/api/profiler/supplement_tracing.h"\n\n#ifdef PADDLE_WITH_DISTRIBUTE\n#include "paddle/phi/infermeta/spmd_rules/rules.h"\n#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"\n#endif\n\nPD_DECLARE_int32(low_precision_op_list);\n'

def api_namespace():
    if False:
        for i in range(10):
            print('nop')
    return ('\nnamespace paddle {\nnamespace experimental {\n\n', '\n\n}  // namespace experimental\n}  // namespace paddle\n')

def sparse_namespace():
    if False:
        return 10
    return ('\nnamespace sparse {\n', '\n}  // namespace sparse\n')

def generate_intermediate_api(api_yaml_path, sparse_api_yaml_path, dygraph_header_file_path, dygraph_source_file_path, gen_dist_branch):
    if False:
        i = 10
        return i + 15
    dygraph_header_file = open(dygraph_header_file_path, 'w')
    dygraph_source_file = open(dygraph_source_file_path, 'w')
    namespace = api_namespace()
    sparse_namespace_pair = sparse_namespace()
    dygraph_header_file.write('#pragma once\n')
    dygraph_header_file.write(header_include())
    dygraph_header_file.write(namespace[0])
    dygraph_include_header_file = 'paddle/phi/api/lib/dygraph_api.h'
    dygraph_source_file.write(source_include(dygraph_include_header_file))
    dygraph_source_file.write(namespace[0])
    apis = []
    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)
    for api in apis:
        foward_api = DistForwardAPI(api) if gen_dist_branch else ForwardAPI(api)
        if foward_api.is_dygraph_api:
            dygraph_header_file.write(foward_api.gene_api_declaration())
            dygraph_source_file.write(foward_api.gene_api_code())
    dygraph_header_file.write(sparse_namespace_pair[0])
    dygraph_source_file.write(sparse_namespace_pair[0])
    with open(sparse_api_yaml_path, 'r') as f:
        sparse_apis = yaml.load(f, Loader=yaml.FullLoader)
    for api in sparse_apis:
        sparse_api = SparseAPI(api)
        if sparse_api.is_dygraph_api:
            dygraph_header_file.write(sparse_api.gene_api_declaration())
            dygraph_source_file.write(sparse_api.gene_api_code())
    dygraph_header_file.write(sparse_namespace_pair[1])
    dygraph_header_file.write(namespace[1])
    dygraph_source_file.write(sparse_namespace_pair[1])
    dygraph_source_file.write(namespace[1])
    dygraph_header_file.close()
    dygraph_source_file.close()

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Generate PaddlePaddle C++ Sparse API files')
    parser.add_argument('--api_yaml_path', nargs='+', help='path to api yaml file', default=['paddle/phi/api/yaml/ops.yaml'])
    parser.add_argument('--sparse_api_yaml_path', help='path to sparse api yaml file', default='paddle/phi/api/yaml/sparse_ops.yaml')
    parser.add_argument('--dygraph_api_header_path', help='output of generated dygraph api header code file', default='paddle/phi/api/lib/dygraph_api.h')
    parser.add_argument('--dygraph_api_source_path', help='output of generated dygraph api source code file', default='paddle/phi/api/lib/dygraph_api.cc')
    parser.add_argument('--gen_dist_branch', help='whether generate distributed branch code', dest='gen_dist_branch', action='store_true')
    options = parser.parse_args()
    api_yaml_path = options.api_yaml_path
    sparse_api_yaml_path = options.sparse_api_yaml_path
    dygraph_header_file_path = options.dygraph_api_header_path
    dygraph_source_file_path = options.dygraph_api_source_path
    gen_dist_branch = options.gen_dist_branch
    generate_intermediate_api(api_yaml_path, sparse_api_yaml_path, dygraph_header_file_path, dygraph_source_file_path, gen_dist_branch)
if __name__ == '__main__':
    main()