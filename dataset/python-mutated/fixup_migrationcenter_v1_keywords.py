import argparse
import os
import libcst as cst
import pathlib
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

def partition(predicate: Callable[[Any], bool], iterator: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    if False:
        i = 10
        return i + 15
    'A stable, out-of-place partition.'
    results = ([], [])
    for i in iterator:
        results[int(predicate(i))].append(i)
    return (results[1], results[0])

class migrationcenterCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'add_assets_to_group': ('group', 'assets', 'request_id', 'allow_existing'), 'aggregate_assets_values': ('parent', 'aggregations', 'filter'), 'batch_delete_assets': ('parent', 'names', 'allow_missing'), 'batch_update_assets': ('parent', 'requests'), 'create_group': ('parent', 'group_id', 'group', 'request_id'), 'create_import_data_file': ('parent', 'import_data_file_id', 'import_data_file', 'request_id'), 'create_import_job': ('parent', 'import_job_id', 'import_job', 'request_id'), 'create_preference_set': ('parent', 'preference_set_id', 'preference_set', 'request_id'), 'create_report': ('parent', 'report_id', 'report', 'request_id'), 'create_report_config': ('parent', 'report_config_id', 'report_config', 'request_id'), 'create_source': ('parent', 'source_id', 'source', 'request_id'), 'delete_asset': ('name', 'request_id'), 'delete_group': ('name', 'request_id'), 'delete_import_data_file': ('name', 'request_id'), 'delete_import_job': ('name', 'request_id', 'force'), 'delete_preference_set': ('name', 'request_id'), 'delete_report': ('name', 'request_id'), 'delete_report_config': ('name', 'request_id', 'force'), 'delete_source': ('name', 'request_id'), 'get_asset': ('name', 'view'), 'get_error_frame': ('name', 'view'), 'get_group': ('name',), 'get_import_data_file': ('name',), 'get_import_job': ('name', 'view'), 'get_preference_set': ('name',), 'get_report': ('name', 'view'), 'get_report_config': ('name',), 'get_settings': ('name',), 'get_source': ('name',), 'list_assets': ('parent', 'page_size', 'page_token', 'filter', 'order_by', 'view'), 'list_error_frames': ('parent', 'page_size', 'page_token', 'view'), 'list_groups': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_import_data_files': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_import_jobs': ('parent', 'page_size', 'page_token', 'filter', 'order_by', 'view'), 'list_preference_sets': ('parent', 'page_size', 'page_token', 'order_by'), 'list_report_configs': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'list_reports': ('parent', 'page_size', 'page_token', 'filter', 'order_by', 'view'), 'list_sources': ('parent', 'page_size', 'page_token', 'filter', 'order_by'), 'remove_assets_from_group': ('group', 'assets', 'request_id', 'allow_missing'), 'report_asset_frames': ('parent', 'source', 'frames'), 'run_import_job': ('name', 'request_id'), 'update_asset': ('update_mask', 'asset', 'request_id'), 'update_group': ('update_mask', 'group', 'request_id'), 'update_import_job': ('update_mask', 'import_job', 'request_id'), 'update_preference_set': ('update_mask', 'preference_set', 'request_id'), 'update_settings': ('update_mask', 'settings', 'request_id'), 'update_source': ('update_mask', 'source', 'request_id'), 'validate_import_job': ('name', 'request_id')}

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.CSTNode:
        if False:
            i = 10
            return i + 15
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

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=migrationcenterCallTransformer()):
    if False:
        for i in range(10):
            print('nop')
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
    parser = argparse.ArgumentParser(description='Fix up source that uses the migrationcenter client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
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