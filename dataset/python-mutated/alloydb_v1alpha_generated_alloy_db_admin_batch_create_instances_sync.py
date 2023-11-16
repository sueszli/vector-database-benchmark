from google.cloud import alloydb_v1alpha

def sample_batch_create_instances():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    requests = alloydb_v1alpha.CreateInstanceRequests()
    requests.create_instance_requests.parent = 'parent_value'
    requests.create_instance_requests.instance_id = 'instance_id_value'
    requests.create_instance_requests.instance.instance_type = 'SECONDARY'
    request = alloydb_v1alpha.BatchCreateInstancesRequest(parent='parent_value', requests=requests)
    operation = client.batch_create_instances(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)