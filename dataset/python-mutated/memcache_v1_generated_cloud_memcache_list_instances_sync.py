from google.cloud import memcache_v1

def sample_list_instances():
    if False:
        i = 10
        return i + 15
    client = memcache_v1.CloudMemcacheClient()
    request = memcache_v1.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)