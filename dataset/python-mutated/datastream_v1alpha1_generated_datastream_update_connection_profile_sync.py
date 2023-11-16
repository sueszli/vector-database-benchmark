from google.cloud import datastream_v1alpha1

def sample_update_connection_profile():
    if False:
        print('Hello World!')
    client = datastream_v1alpha1.DatastreamClient()
    connection_profile = datastream_v1alpha1.ConnectionProfile()
    connection_profile.oracle_profile.hostname = 'hostname_value'
    connection_profile.oracle_profile.username = 'username_value'
    connection_profile.oracle_profile.password = 'password_value'
    connection_profile.oracle_profile.database_service = 'database_service_value'
    connection_profile.display_name = 'display_name_value'
    request = datastream_v1alpha1.UpdateConnectionProfileRequest(connection_profile=connection_profile)
    operation = client.update_connection_profile(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)