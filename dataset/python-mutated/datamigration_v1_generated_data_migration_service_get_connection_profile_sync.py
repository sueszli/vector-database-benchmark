from google.cloud import clouddms_v1

def sample_get_connection_profile():
    if False:
        print('Hello World!')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.GetConnectionProfileRequest(name='name_value')
    response = client.get_connection_profile(request=request)
    print(response)