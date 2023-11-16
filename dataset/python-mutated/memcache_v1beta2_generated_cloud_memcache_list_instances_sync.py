from google.cloud import memcache_v1beta2

def sample_list_instances():
    if False:
        return 10
    client = memcache_v1beta2.CloudMemcacheClient()
    request = memcache_v1beta2.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)