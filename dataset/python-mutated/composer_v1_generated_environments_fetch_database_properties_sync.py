from google.cloud.orchestration.airflow import service_v1

def sample_fetch_database_properties():
    if False:
        return 10
    client = service_v1.EnvironmentsClient()
    request = service_v1.FetchDatabasePropertiesRequest(environment='environment_value')
    response = client.fetch_database_properties(request=request)
    print(response)