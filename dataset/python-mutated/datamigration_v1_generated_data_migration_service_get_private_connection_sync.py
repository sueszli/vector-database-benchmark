from google.cloud import clouddms_v1

def sample_get_private_connection():
    if False:
        while True:
            i = 10
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.GetPrivateConnectionRequest(name='name_value')
    response = client.get_private_connection(request=request)
    print(response)