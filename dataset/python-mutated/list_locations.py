from google.cloud import vmwareengine_v1
from google.cloud.location.locations_pb2 import ListLocationsRequest

def list_locations(project_id: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Prints a list of available locations for use in VMWare Engine.\n\n    Args:\n        project_id: name of the project you want to use.\n\n    Returns:\n        String containing the list of all available locations.\n    '
    client = vmwareengine_v1.VmwareEngineClient()
    request = ListLocationsRequest()
    request.name = f'projects/{project_id}'
    locations = client.list_locations(request)
    print(locations)
    return str(locations)