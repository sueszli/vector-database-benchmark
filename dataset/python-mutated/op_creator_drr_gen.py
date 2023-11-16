import argparse
import yaml
from op_gen import OpCompatParser, OpInfoParser, check_need_update_ops, to_pascal_case, update_ops
CPP_FILE_TEMPLATE = '\n#include "paddle/fluid/pir/drr/ir_operation_factory.h"\n\n{op_header}\n#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"\n\nnamespace pir {{\nnamespace drr {{\n\nvoid OperationFactory::Register{dialect}GeneratedOpCreator() {{\n{body}\n}}\n\n}}  // namespace drr\n}}  // namespace pir\n\n'
NORMAL_FUNCTION_TEMPLATE = '\n  RegisterOperationCreator(\n      "{op_name}",\n      [](const std::vector<Value>& inputs,\n         const pir::AttributeMap& attrs,\n         pir::PatternRewriter& rewriter) {{\n        return rewriter.Build<{namespace}::{op_class_name}>(\n         {params_code});\n      }});\n'
MUTABLE_ATTR_FUNCTION_TEMPLATE = '\n  RegisterOperationCreator(\n      "{op_name}",\n      [](const std::vector<Value>& inputs,\n         const pir::AttributeMap& attrs,\n         pir::PatternRewriter& rewriter) {{\n        // mutable_attr is tensor\n        if (inputs.size() > {inputs_num}) {{\n          return rewriter.Build<paddle::dialect::{op_class_name}>(\n          {params_code_with_mutable_attr});\n        }} else {{\n          return rewriter.Build<paddle::dialect::{op_class_name}>(\n          {params_code_no_mutable_attr});\n        }}\n      }});\n'
Dialect2NameSpaceMap = {'pd_op': 'paddle::dialect', 'cinn_op': 'cinn::dialect'}
Dialect2OpHeaderMap = {'pd_op': '#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"', 'cinn_op': '#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"'}

class OpCreatorCodeGen:

    def __init__(self, op_yaml_files, op_compat_yaml_file, dialect_name):
        if False:
            i = 10
            return i + 15
        self.op_info_items = self.parse_yaml(op_yaml_files, op_compat_yaml_file)
        self.dialect_name = dialect_name

    def parse_yaml(self, op_yaml_files, op_compat_yaml_file):
        if False:
            for i in range(10):
                print('nop')
        op_compat_parser = OpCompatParser(op_compat_yaml_file)
        (need_update_ops, update_yaml_file) = check_need_update_ops(op_yaml_files)
        op_yaml_items = []
        for yaml_file in op_yaml_files:
            if update_yaml_file == yaml_file:
                continue
            with open(yaml_file, 'r') as f:
                ops = yaml.safe_load(f)
                op_yaml_items = op_yaml_items + ops
        if need_update_ops:
            update_ops(op_yaml_items, update_yaml_file)
        op_info_items = []
        for op in op_yaml_items:
            op_compat_item = op_compat_parser.get_compat(op['name'])
            if op_compat_item is not None and op_compat_item['op'] == 'pow' and ('scalar' in op_compat_item):
                op_compat_item = op_compat_item.pop('scalar')
            op_info_items.append(OpInfoParser(op, op_compat_item))
        return op_info_items

    def gen_cpp_file_code(self, cpp_file_path):
        if False:
            print('Hello World!')
        body_code = ''
        for op_info_item in self.op_info_items:
            if op_info_item.infer_meta_map is None:
                continue
            for phi_op_name in op_info_item.op_phi_name:
                ir_op_name = self.dialect_name + '.' + phi_op_name
                params_no_mutable_attr = []
                for i in range(len(op_info_item.input_name_list)):
                    params_no_mutable_attr.append(f'inputs[{i}].dyn_cast<pir::OpResult>()')
                if len(op_info_item.attribute_name_list) > 0:
                    params_no_mutable_attr.append('attrs')
                if self.dialect_name != 'pd_op' or len(op_info_item.mutable_attribute_name_list) == 0:
                    body_code += NORMAL_FUNCTION_TEMPLATE.format(op_name=ir_op_name, namespace=Dialect2NameSpaceMap[self.dialect_name], op_class_name=to_pascal_case(phi_op_name) + 'Op', params_code=', '.join(params_no_mutable_attr))
                else:
                    params_with_mutable_attr = []
                    for i in range(len(op_info_item.input_name_list) + len(op_info_item.mutable_attribute_name_list)):
                        params_with_mutable_attr.append(f'inputs[{i}].dyn_cast<pir::OpResult>()')
                    if len(op_info_item.attribute_name_list) > len(op_info_item.mutable_attribute_name_list):
                        continue
                    body_code += MUTABLE_ATTR_FUNCTION_TEMPLATE.format(op_name=ir_op_name, inputs_num=len(op_info_item.input_name_list), op_class_name=to_pascal_case(phi_op_name) + 'Op', params_code_with_mutable_attr=','.join(params_with_mutable_attr), params_code_no_mutable_attr=', '.join(params_no_mutable_attr))
        with open(cpp_file_path, 'w') as f:
            f.write(CPP_FILE_TEMPLATE.format(dialect=to_pascal_case(self.dialect_name), op_header=Dialect2OpHeaderMap[self.dialect_name], body=body_code))

def ParseArguments():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Generate Op Creator Files By Yaml')
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--dialect_name', type=str)
    parser.add_argument('--op_creator_file', type=str)
    return parser.parse_args()
if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(',')
    op_compat_yaml_file = args.op_compat_yaml_file
    op_creator_file = args.op_creator_file
    dialect_name = args.dialect_name
    code_gen = OpCreatorCodeGen(op_yaml_files, op_compat_yaml_file, dialect_name)
    code_gen.gen_cpp_file_code(op_creator_file)