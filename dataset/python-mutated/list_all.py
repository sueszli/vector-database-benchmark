from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from google.cloud import compute_v1

def list_all_instances(project_id: str) -> dict[str, Iterable[compute_v1.Instance]]:
    if False:
        i = 10
        return i + 15
    '\n    Returns a dictionary of all instances present in a project, grouped by their zone.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n    Returns:\n        A dictionary with zone names as keys (in form of "zones/{zone_name}") and\n        iterable collections of Instance objects as values.\n    '
    instance_client = compute_v1.InstancesClient()
    request = compute_v1.AggregatedListInstancesRequest()
    request.project = project_id
    request.max_results = 50
    agg_list = instance_client.aggregated_list(request=request)
    all_instances = defaultdict(list)
    print('Instances found:')
    for (zone, response) in agg_list:
        if response.instances:
            all_instances[zone].extend(response.instances)
            print(f' {zone}:')
            for instance in response.instances:
                print(f' - {instance.name} ({instance.machine_type})')
    return all_instances