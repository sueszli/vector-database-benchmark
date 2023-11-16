from google.cloud import apigeeconnect_v1

async def sample_egress():
    client = apigeeconnect_v1.TetherAsyncClient()
    request = apigeeconnect_v1.EgressResponse()
    requests = [request]

    def request_generator():
        if False:
            for i in range(10):
                print('nop')
        for request in requests:
            yield request
    stream = await client.egress(requests=request_generator())
    async for response in stream:
        print(response)