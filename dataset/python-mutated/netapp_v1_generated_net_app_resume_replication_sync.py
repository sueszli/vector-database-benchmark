from google.cloud import netapp_v1

def sample_resume_replication():
    if False:
        return 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.ResumeReplicationRequest(name='name_value')
    operation = client.resume_replication(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)