from google.cloud.orchestration.airflow import service_v1beta1

def sample_check_upgrade():
    if False:
        for i in range(10):
            print('nop')
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.CheckUpgradeRequest()
    operation = client.check_upgrade(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)