import argparse
import os
import libcst as cst
import pathlib
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

def partition(predicate: Callable[[Any], bool], iterator: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    if False:
        while True:
            i = 10
    'A stable, out-of-place partition.'
    results = ([], [])
    for i in iterator:
        results[int(predicate(i))].append(i)
    return (results[1], results[0])

class dataflowCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'aggregated_list_jobs': ('filter', 'project_id', 'view', 'page_size', 'page_token', 'location'), 'check_active_jobs': ('project_id',), 'create_job': ('project_id', 'job', 'view', 'replace_job_id', 'location'), 'create_job_from_template': ('project_id', 'job_name', 'gcs_path', 'parameters', 'environment', 'location'), 'delete_snapshot': ('project_id', 'snapshot_id', 'location'), 'get_job': ('project_id', 'job_id', 'view', 'location'), 'get_job_execution_details': ('project_id', 'job_id', 'location', 'page_size', 'page_token'), 'get_job_metrics': ('project_id', 'job_id', 'start_time', 'location'), 'get_snapshot': ('project_id', 'snapshot_id', 'location'), 'get_stage_execution_details': ('project_id', 'job_id', 'location', 'stage_id', 'page_size', 'page_token', 'start_time', 'end_time'), 'get_template': ('project_id', 'gcs_path', 'view', 'location'), 'launch_flex_template': ('project_id', 'launch_parameter', 'location', 'validate_only'), 'launch_template': ('project_id', 'validate_only', 'gcs_path', 'dynamic_template', 'launch_parameters', 'location'), 'list_job_messages': ('project_id', 'job_id', 'minimum_importance', 'page_size', 'page_token', 'start_time', 'end_time', 'location'), 'list_jobs': ('filter', 'project_id', 'view', 'page_size', 'page_token', 'location'), 'list_snapshots': ('project_id', 'job_id', 'location'), 'snapshot_job': ('project_id', 'job_id', 'ttl', 'location', 'snapshot_sources', 'description'), 'update_job': ('project_id', 'job_id', 'job', 'location')}

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

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=dataflowCallTransformer()):
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
    parser = argparse.ArgumentParser(description='Fix up source that uses the dataflow client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
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