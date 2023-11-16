import argparse
import os
import libcst as cst
import pathlib
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

def partition(predicate: Callable[[Any], bool], iterator: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    if False:
        print('Hello World!')
    'A stable, out-of-place partition.'
    results = ([], [])
    for i in iterator:
        results[int(predicate(i))].append(i)
    return (results[1], results[0])

class bare_metal_solutionCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'create_nfs_share': ('parent', 'nfs_share'), 'create_provisioning_config': ('parent', 'provisioning_config', 'email'), 'create_ssh_key': ('parent', 'ssh_key', 'ssh_key_id'), 'create_volume_snapshot': ('parent', 'volume_snapshot'), 'delete_nfs_share': ('name',), 'delete_ssh_key': ('name',), 'delete_volume_snapshot': ('name',), 'detach_lun': ('instance', 'lun', 'skip_reboot'), 'disable_interactive_serial_console': ('name',), 'enable_interactive_serial_console': ('name',), 'evict_lun': ('name',), 'evict_volume': ('name',), 'get_instance': ('name',), 'get_lun': ('name',), 'get_network': ('name',), 'get_nfs_share': ('name',), 'get_provisioning_config': ('name',), 'get_volume': ('name',), 'get_volume_snapshot': ('name',), 'list_instances': ('parent', 'page_size', 'page_token', 'filter'), 'list_luns': ('parent', 'page_size', 'page_token'), 'list_networks': ('parent', 'page_size', 'page_token', 'filter'), 'list_network_usage': ('location',), 'list_nfs_shares': ('parent', 'page_size', 'page_token', 'filter'), 'list_os_images': ('parent', 'page_size', 'page_token'), 'list_provisioning_quotas': ('parent', 'page_size', 'page_token'), 'list_ssh_keys': ('parent', 'page_size', 'page_token'), 'list_volumes': ('parent', 'page_size', 'page_token', 'filter'), 'list_volume_snapshots': ('parent', 'page_size', 'page_token'), 'rename_instance': ('name', 'new_instance_id'), 'rename_network': ('name', 'new_network_id'), 'rename_nfs_share': ('name', 'new_nfsshare_id'), 'rename_volume': ('name', 'new_volume_id'), 'reset_instance': ('name',), 'resize_volume': ('volume', 'size_gib'), 'restore_volume_snapshot': ('volume_snapshot',), 'start_instance': ('name',), 'stop_instance': ('name',), 'submit_provisioning_config': ('parent', 'provisioning_config', 'email'), 'update_instance': ('instance', 'update_mask'), 'update_network': ('network', 'update_mask'), 'update_nfs_share': ('nfs_share', 'update_mask'), 'update_provisioning_config': ('provisioning_config', 'update_mask', 'email'), 'update_volume': ('volume', 'update_mask')}

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

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=bare_metal_solutionCallTransformer()):
    if False:
        while True:
            i = 10
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
    parser = argparse.ArgumentParser(description='Fix up source that uses the bare_metal_solution client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
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