from google.cloud.orchestration.airflow import service_v1beta1

def sample_fetch_database_properties():
    if False:
        for i in range(10):
            print('nop')
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.FetchDatabasePropertiesRequest(environment='environment_value')
    response = client.fetch_database_properties(request=request)
    print(response)