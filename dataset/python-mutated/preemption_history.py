from __future__ import annotations
import datetime
from google.cloud import compute_v1
from google.cloud.compute_v1.services.zone_operations import pagers

def list_zone_operations(project_id: str, zone: str, filter: str='') -> pagers.ListPager:
    if False:
        i = 10
        return i + 15
    '\n    List all recent operations the happened in given zone in a project. Optionally filter those\n    operations by providing a filter. More about using the filter can be found here:\n    https://cloud.google.com/compute/docs/reference/rest/v1/zoneOperations/list\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone you want to use. For example: "us-west3-b"\n        filter: filter string to be used for this listing operation.\n    Returns:\n        List of preemption operations in given zone.\n    '
    operation_client = compute_v1.ZoneOperationsClient()
    request = compute_v1.ListZoneOperationsRequest()
    request.project = project_id
    request.zone = zone
    request.filter = filter
    return operation_client.list(request)

def preemption_history(project_id: str, zone: str, instance_name: str=None) -> list[tuple[str, datetime.datetime]]:
    if False:
        print('Hello World!')
    '\n    Get a list of preemption operations from given zone in a project. Optionally limit\n    the results to instance name.\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone you want to use. For example: "us-west3-b"\n        instance_name: name of the virtual machine to look for.\n    Returns:\n        List of preemption operations in given zone.\n    '
    if instance_name:
        filter = f'operationType="compute.instances.preempted" AND targetLink:instances/{instance_name}'
    else:
        filter = 'operationType="compute.instances.preempted"'
    history = []
    for operation in list_zone_operations(project_id, zone, filter):
        this_instance_name = operation.target_link.rsplit('/', maxsplit=1)[1]
        if instance_name and this_instance_name == instance_name:
            moment = datetime.datetime.fromisoformat(operation.insert_time)
            history.append((instance_name, moment))
    return history