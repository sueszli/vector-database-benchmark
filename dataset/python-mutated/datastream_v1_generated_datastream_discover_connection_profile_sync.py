from google.cloud import datastream_v1

def sample_discover_connection_profile():
    if False:
        for i in range(10):
            print('nop')
    client = datastream_v1.DatastreamClient()
    connection_profile = datastream_v1.ConnectionProfile()
    connection_profile.oracle_profile.hostname = 'hostname_value'
    connection_profile.oracle_profile.username = 'username_value'
    connection_profile.oracle_profile.password = 'password_value'
    connection_profile.oracle_profile.database_service = 'database_service_value'
    connection_profile.display_name = 'display_name_value'
    request = datastream_v1.DiscoverConnectionProfileRequest(connection_profile=connection_profile, full_hierarchy=True, parent='parent_value')
    response = client.discover_connection_profile(request=request)
    print(response)