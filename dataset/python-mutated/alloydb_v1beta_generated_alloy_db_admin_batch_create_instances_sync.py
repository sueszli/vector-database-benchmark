from google.cloud import alloydb_v1beta

def sample_batch_create_instances():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1beta.AlloyDBAdminClient()
    requests = alloydb_v1beta.CreateInstanceRequests()
    requests.create_instance_requests.parent = 'parent_value'
    requests.create_instance_requests.instance_id = 'instance_id_value'
    requests.create_instance_requests.instance.instance_type = 'SECONDARY'
    request = alloydb_v1beta.BatchCreateInstancesRequest(parent='parent_value', requests=requests)
    operation = client.batch_create_instances(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)