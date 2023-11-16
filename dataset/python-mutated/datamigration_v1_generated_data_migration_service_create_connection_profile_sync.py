from google.cloud import clouddms_v1

def sample_create_connection_profile():
    if False:
        i = 10
        return i + 15
    client = clouddms_v1.DataMigrationServiceClient()
    connection_profile = clouddms_v1.ConnectionProfile()
    connection_profile.mysql.host = 'host_value'
    connection_profile.mysql.port = 453
    connection_profile.mysql.username = 'username_value'
    connection_profile.mysql.password = 'password_value'
    request = clouddms_v1.CreateConnectionProfileRequest(parent='parent_value', connection_profile_id='connection_profile_id_value', connection_profile=connection_profile)
    operation = client.create_connection_profile(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)