from google.cloud.orchestration.airflow import service_v1

def sample_database_failover():
    if False:
        for i in range(10):
            print('nop')
    client = service_v1.EnvironmentsClient()
    request = service_v1.DatabaseFailoverRequest()
    operation = client.database_failover(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)