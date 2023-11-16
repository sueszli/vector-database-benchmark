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

class securitycenterCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'create_finding': ('parent', 'finding_id', 'finding'), 'create_notification_config': ('parent', 'config_id', 'notification_config'), 'create_source': ('parent', 'source'), 'delete_notification_config': ('name',), 'get_iam_policy': ('resource', 'options'), 'get_notification_config': ('name',), 'get_organization_settings': ('name',), 'get_source': ('name',), 'group_assets': ('parent', 'group_by', 'filter', 'compare_duration', 'read_time', 'page_token', 'page_size'), 'group_findings': ('parent', 'group_by', 'filter', 'read_time', 'compare_duration', 'page_token', 'page_size'), 'list_assets': ('parent', 'filter', 'order_by', 'read_time', 'compare_duration', 'field_mask', 'page_token', 'page_size'), 'list_findings': ('parent', 'filter', 'order_by', 'read_time', 'compare_duration', 'field_mask', 'page_token', 'page_size'), 'list_notification_configs': ('parent', 'page_token', 'page_size'), 'list_sources': ('parent', 'page_token', 'page_size'), 'run_asset_discovery': ('parent',), 'set_finding_state': ('name', 'state', 'start_time'), 'set_iam_policy': ('resource', 'policy', 'update_mask'), 'test_iam_permissions': ('resource', 'permissions'), 'update_finding': ('finding', 'update_mask'), 'update_notification_config': ('notification_config', 'update_mask'), 'update_organization_settings': ('organization_settings', 'update_mask'), 'update_security_marks': ('security_marks', 'update_mask', 'start_time'), 'update_source': ('source', 'update_mask')}

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

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=securitycenterCallTransformer()):
    if False:
        print('Hello World!')
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
    parser = argparse.ArgumentParser(description='Fix up source that uses the securitycenter client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
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