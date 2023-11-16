from google.api_core.exceptions import GoogleAPICallError
from google.cloud import optimization_v1

def call_async_api(project_id: str, request_model_gcs_path: str, model_solution_gcs_path_prefix: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Call the async api for fleet routing.'
    fleet_routing_client = optimization_v1.FleetRoutingClient()
    request_file_name = 'resources/async_request.json'
    with open(request_file_name) as f:
        fleet_routing_request = optimization_v1.BatchOptimizeToursRequest.from_json(f.read())
        fleet_routing_request.parent = f'projects/{project_id}'
        for (idx, mc) in enumerate(fleet_routing_request.model_configs):
            mc.input_config.gcs_source.uri = request_model_gcs_path
            model_solution_gcs_path = f'{model_solution_gcs_path_prefix}_{idx}'
            mc.output_config.gcs_destination.uri = model_solution_gcs_path
        operation = fleet_routing_client.batch_optimize_tours(fleet_routing_request)
        print(operation.operation.name)
        try:
            result = operation.result()
            print(result)
        except GoogleAPICallError:
            print(operation.operation.error)