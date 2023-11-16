from google.cloud import apigeeconnect_v1

def sample_egress():
    if False:
        return 10
    client = apigeeconnect_v1.TetherClient()
    request = apigeeconnect_v1.EgressResponse()
    requests = [request]

    def request_generator():
        if False:
            return 10
        for request in requests:
            yield request
    stream = client.egress(requests=request_generator())
    for response in stream:
        print(response)