from google.cloud import compute_v1
from google.cloud.compute_v1.services.zone_operations import pagers

def list_zone_operations(project_id: str, zone: str, filter: str='') -> pagers.ListPager:
    if False:
        while True:
            i = 10
    '\n    List all recent operations the happened in given zone in a project. Optionally filter those\n    operations by providing a filter. More about using the filter can be found here:\n    https://cloud.google.com/compute/docs/reference/rest/v1/zoneOperations/list\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone you want to use. For example: "us-west3-b"\n        filter: filter string to be used for this listing operation.\n    Returns:\n        List of preemption operations in given zone.\n    '
    operation_client = compute_v1.ZoneOperationsClient()
    request = compute_v1.ListZoneOperationsRequest()
    request.project = project_id
    request.zone = zone
    request.filter = filter
    return operation_client.list(request)