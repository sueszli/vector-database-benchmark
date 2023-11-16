import logging
from typing import Dict
from twisted.web.resource import Resource
from synapse.http.server import UnrecognizedRequestResource
logger = logging.getLogger(__name__)

def create_resource_tree(desired_tree: Dict[str, Resource], root_resource: Resource) -> Resource:
    if False:
        for i in range(10):
            print('nop')
    'Create the resource tree for this homeserver.\n\n    This in unduly complicated because Twisted does not support putting\n    child resources more than 1 level deep at a time.\n\n    Args:\n        desired_tree: Dict from desired paths to desired resources.\n        root_resource: The root resource to add the tree to.\n    Returns:\n        The ``root_resource`` with a tree of child resources added to it.\n    '
    resource_mappings: Dict[str, Resource] = {}
    for (full_path_str, res) in desired_tree.items():
        full_path = full_path_str.encode('utf-8')
        logger.info('Attaching %s to path %s', res, full_path)
        last_resource = root_resource
        for path_seg in full_path.split(b'/')[1:-1]:
            if path_seg not in last_resource.listNames():
                child_resource: Resource = UnrecognizedRequestResource()
                last_resource.putChild(path_seg, child_resource)
                res_id = _resource_id(last_resource, path_seg)
                resource_mappings[res_id] = child_resource
                last_resource = child_resource
            else:
                res_id = _resource_id(last_resource, path_seg)
                last_resource = resource_mappings[res_id]
        last_path_seg = full_path.split(b'/')[-1]
        res_id = _resource_id(last_resource, last_path_seg)
        if res_id in resource_mappings:
            existing_dummy_resource = resource_mappings[res_id]
            for child_name in existing_dummy_resource.listNames():
                child_res_id = _resource_id(existing_dummy_resource, child_name)
                child_resource = resource_mappings[child_res_id]
                res.putChild(child_name, child_resource)
        last_resource.putChild(last_path_seg, res)
        res_id = _resource_id(last_resource, last_path_seg)
        resource_mappings[res_id] = res
    return root_resource

def _resource_id(resource: Resource, path_seg: bytes) -> str:
    if False:
        print('Hello World!')
    'Construct an arbitrary resource ID so you can retrieve the mapping\n    later.\n\n    If you want to represent resource A putChild resource B with path C,\n    the mapping should looks like _resource_id(A,C) = B.\n\n    Args:\n        resource: The *parent* Resourceb\n        path_seg: The name of the child Resource to be attached.\n    Returns:\n        A unique string which can be a key to the child Resource.\n    '
    return '%s-%r' % (resource, path_seg)