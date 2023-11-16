from google.cloud import datastream_v1

def sample_fetch_static_ips():
    if False:
        print('Hello World!')
    client = datastream_v1.DatastreamClient()
    request = datastream_v1.FetchStaticIpsRequest(name='name_value')
    page_result = client.fetch_static_ips(request=request)
    for response in page_result:
        print(response)