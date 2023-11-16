import argparse
import os
import libcst as cst
import pathlib
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

def partition(predicate: Callable[[Any], bool], iterator: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    if False:
        return 10
    'A stable, out-of-place partition.'
    results = ([], [])
    for i in iterator:
        results[int(predicate(i))].append(i)
    return (results[1], results[0])

class dataplexCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'cancel_job': ('name',), 'create_asset': ('parent', 'asset_id', 'asset', 'validate_only'), 'create_content': ('parent', 'content', 'validate_only'), 'create_data_attribute': ('parent', 'data_attribute_id', 'data_attribute', 'validate_only'), 'create_data_attribute_binding': ('parent', 'data_attribute_binding_id', 'data_attribute_binding', 'validate_only'), 'create_data_scan': ('parent', 'data_scan', 'data_scan_id', 'validate_only'), 'create_data_taxonomy': ('parent', 'data_taxonomy_id', 'data_taxonomy', 'validate_only'), 'create_entity': ('parent', 'entity', 'validate_only'), 'create_environment': ('parent', 'environment_id', 'environment', 'validate_only'), 'create_lake': ('parent', 'lake_id', 'lake', 'validate_only'), 'create_partition': ('parent', 'partition', 'validate_only'), 'create_task': ('parent', 'task_id', 'task', 'validate_only'), 'create_zone': ('parent', 'zone_id', 'zone', 'validate_only'), 'delete_asset': ('name',), 'delete_content': ('name',), 'delete_data_attribute': ('name', 'etag'), 'delete_data_attribute_binding': ('name', 'etag'), 'delete_data_scan': ('name',), 'delete_data_taxonomy': ('name', 'etag'), 'delete_entity': ('name', 'etag'), 'delete_environment': ('name',), 'delete_lake': ('name',), 'delete_partition': ('name', 'etag'), 'delete_task': ('name',), 'delete_zone': ('name',), 'get_asset': ('name',), 'get_content': ('name', 'view'), 'get_data_attribute': ('name',), 'get_data_attribute_binding': ('name',), 'get_data_scan': ('name', 'view'), 'get_data_scan_job': ('name', 'view'), 'get_data_taxonomy': ('name',), 'get_entity': ('name', 'view'), 'get_environment': ('name',), 'get_iam_policy': ('resource', 'options'), 'get_job': ('name',), 'get_lake': ('name',), 'get_partition': ('name',), 'get_task': ('name',), 'get_zone': ('name',), 'list_asset_actions': ('parent', 'page_size', 'page_token'), 'list_assets': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_content': ('parent', 'page_size', 'page_token', 'filter'), 'list_data_attribute_bindings': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_data_attributes': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_data_scan_jobs': ('parent', 'page_size', 'page_token', 'filter'), 'list_data_scans': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_data_taxonomies': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_entities': ('parent', 'view', 'page_size', 'page_token', 'filter'), 'list_environments': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_jobs': ('parent', 'page_size', 'page_token'), 'list_lake_actions': ('parent', 'page_size', 'page_token'), 'list_lakes': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_partitions': ('parent', 'page_size', 'page_token', 'filter'), 'list_sessions': ('parent', 'page_size', 'page_token', 'filter'), 'list_tasks': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_zone_actions': ('parent', 'page_size', 'page_token'), 'list_zones': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'run_data_scan': ('name',), 'run_task': ('name', 'labels', 'args'), 'set_iam_policy': ('resource', 'policy', 'update_mask'), 'test_iam_permissions': ('resource', 'permissions'), 'update_asset': ('update_mask', 'asset', 'validate_only'), 'update_content': ('update_mask', 'content', 'validate_only'), 'update_data_attribute': ('update_mask', 'data_attribute', 'validate_only'), 'update_data_attribute_binding': ('update_mask', 'data_attribute_binding', 'validate_only'), 'update_data_scan': ('data_scan', 'update_mask', 'validate_only'), 'update_data_taxonomy': ('update_mask', 'data_taxonomy', 'validate_only'), 'update_entity': ('entity', 'validate_only'), 'update_environment': ('update_mask', 'environment', 'validate_only'), 'update_lake': ('update_mask', 'lake', 'validate_only'), 'update_task': ('update_mask', 'task', 'validate_only'), 'update_zone': ('update_mask', 'zone', 'validate_only')}

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.CSTNode:
        if False:
            for i in range(10):
                print('nop')
        try:
            key = original.func.attr.value
            kword_params = self.METHOD_TO_PARAMS[key]
        except (AttributeError, KeyError):
            return updated
        (args, kwargs) = partition(lambda a: not bool(a.keyword), updated.args)
        if any((k.keyword.value == 'request' for k in kwargs)):
            return updated
        (kwargs, ctrl_kwargs) = partition(lambda a: a.keyword.value not in self.CTRL_PARAMS, kwargs)
        (args, ctrl_args) = (args[:len(kword_params)], args[len(kword_params):])
        ctrl_kwargs.extend((cst.Arg(value=a.value, keyword=cst.Name(value=ctrl)) for (a, ctrl) in zip(ctrl_args, self.CTRL_PARAMS)))
        request_arg = cst.Arg(value=cst.Dict([cst.DictElement(cst.SimpleString("'{}'".format(name)), cst.Element(value=arg.value)) for (name, arg) in zip(kword_params, args + kwargs)]), keyword=cst.Name('request'))
        return updated.with_changes(args=[request_arg] + ctrl_kwargs)

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=dataplexCallTransformer()):
    if False:
        i = 10
        return i + 15
    'Duplicate the input dir to the output dir, fixing file method calls.\n\n    Preconditions:\n    * in_dir is a real directory\n    * out_dir is a real, empty directory\n    '
    pyfile_gen = (pathlib.Path(os.path.join(root, f)) for (root, _, files) in os.walk(in_dir) for f in files if os.path.splitext(f)[1] == '.py')
    for fpath in pyfile_gen:
        with open(fpath, 'r') as f:
            src = f.read()
        tree = cst.parse_module(src)
        updated = tree.visit(transformer)
        updated_path = out_dir.joinpath(fpath.relative_to(in_dir))
        updated_path.parent.mkdir(parents=True, exist_ok=True)
        with open(updated_path, 'w') as f:
            f.write(updated.code)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix up source that uses the dataplex client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
    parser.add_argument('-d', '--input-directory', required=True, dest='input_dir', help='the input directory to walk for python files to fix up')
    parser.add_argument('-o', '--output-directory', required=True, dest='output_dir', help='the directory to output files fixed via un-flattening')
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"input directory '{input_dir}' does not exist or is not a directory", file=sys.stderr)
        sys.exit(-1)
    if not output_dir.is_dir():
        print(f"output directory '{output_dir}' does not exist or is not a directory", file=sys.stderr)
        sys.exit(-1)
    if os.listdir(output_dir):
        print(f"output directory '{output_dir}' is not empty", file=sys.stderr)
        sys.exit(-1)
    fix_files(input_dir, output_dir)