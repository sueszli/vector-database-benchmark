from google.api import httpbody_pb2
from google.cloud.gkeconnect import gateway_v1beta1

def sample_delete_resource():
    if False:
        return 10
    client = gateway_v1beta1.GatewayServiceClient()
    request = httpbody_pb2.HttpBody()
    response = client.delete_resource(request=request)
    print(response)