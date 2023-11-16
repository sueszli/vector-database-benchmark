from google.cloud import optimization_v1
from google.cloud.optimization_v1.services import fleet_routing
from google.cloud.optimization_v1.services.fleet_routing import transports
from google.cloud.optimization_v1.services.fleet_routing.transports import grpc as fleet_routing_grpc

def long_timeout(request_file_name: str, project_id: str) -> None:
    if False:
        print('Hello World!')
    with open(request_file_name) as f:
        fleet_routing_request = optimization_v1.OptimizeToursRequest.from_json(f.read())
        fleet_routing_request.parent = f'projects/{project_id}'
    channel = fleet_routing_grpc.FleetRoutingGrpcTransport.create_channel(options=[('grpc.keepalive_time_ms', 500), ('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
    transport = transports.FleetRoutingGrpcTransport(channel=channel)
    client = fleet_routing.client.FleetRoutingClient(transport=transport)
    fleet_routing_response = client.optimize_tours(fleet_routing_request)
    print(fleet_routing_response)