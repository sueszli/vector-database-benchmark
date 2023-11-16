from google.cloud import optimization_v1

def call_sync_api(project_id: str) -> None:
    if False:
        while True:
            i = 10
    'Call the sync api for fleet routing.'
    request_file_name = 'resources/sync_request.json'
    fleet_routing_client = optimization_v1.FleetRoutingClient()
    with open(request_file_name) as f:
        fleet_routing_request = optimization_v1.OptimizeToursRequest.from_json(f.read())
        fleet_routing_request.parent = f'projects/{project_id}'
        fleet_routing_response = fleet_routing_client.optimize_tours(fleet_routing_request, timeout=100)
        print(fleet_routing_response)