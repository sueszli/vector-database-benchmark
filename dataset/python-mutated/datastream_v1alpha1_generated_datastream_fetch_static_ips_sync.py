from google.cloud import datastream_v1alpha1

def sample_fetch_static_ips():
    if False:
        return 10
    client = datastream_v1alpha1.DatastreamClient()
    request = datastream_v1alpha1.FetchStaticIpsRequest(name='name_value')
    page_result = client.fetch_static_ips(request=request)
    for response in page_result:
        print(response)