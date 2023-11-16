from google.api import httpbody_pb2
from google.cloud.gkeconnect import gateway_v1beta1

def sample_patch_resource():
    if False:
        for i in range(10):
            print('nop')
    client = gateway_v1beta1.GatewayServiceClient()
    request = httpbody_pb2.HttpBody()
    response = client.patch_resource(request=request)
    print(response)