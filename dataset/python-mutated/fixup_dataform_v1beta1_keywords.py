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

class dataformCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'cancel_workflow_invocation': ('name',), 'commit_repository_changes': ('name', 'commit_metadata', 'required_head_commit_sha', 'file_operations'), 'commit_workspace_changes': ('name', 'author', 'commit_message', 'paths'), 'compute_repository_access_token_status': ('name',), 'create_compilation_result': ('parent', 'compilation_result'), 'create_release_config': ('parent', 'release_config', 'release_config_id'), 'create_repository': ('parent', 'repository', 'repository_id'), 'create_workflow_config': ('parent', 'workflow_config', 'workflow_config_id'), 'create_workflow_invocation': ('parent', 'workflow_invocation'), 'create_workspace': ('parent', 'workspace', 'workspace_id'), 'delete_release_config': ('name',), 'delete_repository': ('name', 'force'), 'delete_workflow_config': ('name',), 'delete_workflow_invocation': ('name',), 'delete_workspace': ('name',), 'fetch_file_diff': ('workspace', 'path'), 'fetch_file_git_statuses': ('name',), 'fetch_git_ahead_behind': ('name', 'remote_branch'), 'fetch_remote_branches': ('name',), 'fetch_repository_history': ('name', 'page_size', 'page_token'), 'get_compilation_result': ('name',), 'get_release_config': ('name',), 'get_repository': ('name',), 'get_workflow_config': ('name',), 'get_workflow_invocation': ('name',), 'get_workspace': ('name',), 'install_npm_packages': ('workspace',), 'list_compilation_results': ('parent', 'page_size', 'page_token'), 'list_release_configs': ('parent', 'page_size', 'page_token'), 'list_repositories': ('parent', 'page_size', 'page_token', 'order_by', 'filter'), 'list_workflow_configs': ('parent', 'page_size', 'page_token'), 'list_workflow_invocations': ('parent', 'page_size', 'page_token', 'order_by', 'filter'), 'list_workspaces': ('parent', 'page_size', 'page_token', 'order_by', 'filter'), 'make_directory': ('workspace', 'path'), 'move_directory': ('workspace', 'path', 'new_path'), 'move_file': ('workspace', 'path', 'new_path'), 'pull_git_commits': ('name', 'author', 'remote_branch'), 'push_git_commits': ('name', 'remote_branch'), 'query_compilation_result_actions': ('name', 'page_size', 'page_token', 'filter'), 'query_directory_contents': ('workspace', 'path', 'page_size', 'page_token'), 'query_repository_directory_contents': ('name', 'commit_sha', 'path', 'page_size', 'page_token'), 'query_workflow_invocation_actions': ('name', 'page_size', 'page_token'), 'read_file': ('workspace', 'path'), 'read_repository_file': ('name', 'path', 'commit_sha'), 'remove_directory': ('workspace', 'path'), 'remove_file': ('workspace', 'path'), 'reset_workspace_changes': ('name', 'paths', 'clean'), 'update_release_config': ('release_config', 'update_mask'), 'update_repository': ('repository', 'update_mask'), 'update_workflow_config': ('workflow_config', 'update_mask'), 'write_file': ('workspace', 'path', 'contents')}

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.CSTNode:
        if False:
            while True:
                i = 10
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

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=dataformCallTransformer()):
    if False:
        return 10
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
    parser = argparse.ArgumentParser(description='Fix up source that uses the dataform client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
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