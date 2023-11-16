import argparse
import os
import libcst as cst
import pathlib
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

def partition(predicate: Callable[[Any], bool], iterator: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    if False:
        for i in range(10):
            print('nop')
    'A stable, out-of-place partition.'
    results = ([], [])
    for i in iterator:
        results[int(predicate(i))].append(i)
    return (results[1], results[0])

class containerCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {'cancel_operation': ('project_id', 'zone', 'operation_id', 'name'), 'check_autopilot_compatibility': ('name',), 'complete_ip_rotation': ('project_id', 'zone', 'cluster_id', 'name'), 'complete_node_pool_upgrade': ('name',), 'create_cluster': ('cluster', 'project_id', 'zone', 'parent'), 'create_node_pool': ('node_pool', 'project_id', 'zone', 'cluster_id', 'parent'), 'delete_cluster': ('project_id', 'zone', 'cluster_id', 'name'), 'delete_node_pool': ('project_id', 'zone', 'cluster_id', 'node_pool_id', 'name'), 'get_cluster': ('project_id', 'zone', 'cluster_id', 'name'), 'get_json_web_keys': ('parent',), 'get_node_pool': ('project_id', 'zone', 'cluster_id', 'node_pool_id', 'name'), 'get_operation': ('project_id', 'zone', 'operation_id', 'name'), 'get_server_config': ('project_id', 'zone', 'name'), 'list_clusters': ('project_id', 'zone', 'parent'), 'list_node_pools': ('project_id', 'zone', 'cluster_id', 'parent'), 'list_operations': ('project_id', 'zone', 'parent'), 'list_usable_subnetworks': ('parent', 'filter', 'page_size', 'page_token'), 'rollback_node_pool_upgrade': ('project_id', 'zone', 'cluster_id', 'node_pool_id', 'name', 'respect_pdb'), 'set_addons_config': ('addons_config', 'project_id', 'zone', 'cluster_id', 'name'), 'set_labels': ('resource_labels', 'label_fingerprint', 'project_id', 'zone', 'cluster_id', 'name'), 'set_legacy_abac': ('enabled', 'project_id', 'zone', 'cluster_id', 'name'), 'set_locations': ('locations', 'project_id', 'zone', 'cluster_id', 'name'), 'set_logging_service': ('logging_service', 'project_id', 'zone', 'cluster_id', 'name'), 'set_maintenance_policy': ('project_id', 'zone', 'cluster_id', 'maintenance_policy', 'name'), 'set_master_auth': ('action', 'update', 'project_id', 'zone', 'cluster_id', 'name'), 'set_monitoring_service': ('monitoring_service', 'project_id', 'zone', 'cluster_id', 'name'), 'set_network_policy': ('network_policy', 'project_id', 'zone', 'cluster_id', 'name'), 'set_node_pool_autoscaling': ('autoscaling', 'project_id', 'zone', 'cluster_id', 'node_pool_id', 'name'), 'set_node_pool_management': ('management', 'project_id', 'zone', 'cluster_id', 'node_pool_id', 'name'), 'set_node_pool_size': ('node_count', 'project_id', 'zone', 'cluster_id', 'node_pool_id', 'name'), 'start_ip_rotation': ('project_id', 'zone', 'cluster_id', 'name', 'rotate_credentials'), 'update_cluster': ('update', 'project_id', 'zone', 'cluster_id', 'name'), 'update_master': ('master_version', 'project_id', 'zone', 'cluster_id', 'name'), 'update_node_pool': ('node_version', 'image_type', 'project_id', 'zone', 'cluster_id', 'node_pool_id', 'name', 'locations', 'workload_metadata_config', 'upgrade_settings', 'tags', 'taints', 'labels', 'linux_node_config', 'kubelet_config', 'node_network_config', 'gcfs_config', 'confidential_nodes', 'gvnic', 'etag', 'fast_socket', 'logging_config', 'resource_labels', 'windows_node_config', 'machine_type', 'disk_type', 'disk_size_gb', 'resource_manager_tags')}

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

def fix_files(in_dir: pathlib.Path, out_dir: pathlib.Path, *, transformer=containerCallTransformer()):
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
    parser = argparse.ArgumentParser(description='Fix up source that uses the container client library.\n\nThe existing sources are NOT overwritten but are copied to output_dir with changes made.\n\nNote: This tool operates at a best-effort level at converting positional\n      parameters in client method calls to keyword based parameters.\n      Cases where it WILL FAIL include\n      A) * or ** expansion in a method call.\n      B) Calls via function or method alias (includes free function calls)\n      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)\n\n      These all constitute false negatives. The tool will also detect false\n      positives when an API method shares a name with another method.\n')
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