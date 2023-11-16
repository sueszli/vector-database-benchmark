from google.cloud import redis_v1

def sample_list_instances():
    if False:
        for i in range(10):
            print('nop')
    client = redis_v1.CloudRedisClient()
    request = redis_v1.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)