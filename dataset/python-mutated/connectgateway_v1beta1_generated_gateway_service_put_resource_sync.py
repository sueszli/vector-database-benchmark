from google.api import httpbody_pb2
from google.cloud.gkeconnect import gateway_v1beta1

def sample_put_resource():
    if False:
        print('Hello World!')
    client = gateway_v1beta1.GatewayServiceClient()
    request = httpbody_pb2.HttpBody()
    response = client.put_resource(request=request)
    print(response)