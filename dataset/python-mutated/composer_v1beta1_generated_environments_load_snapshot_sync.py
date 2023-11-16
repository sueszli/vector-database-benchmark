from google.cloud.orchestration.airflow import service_v1beta1

def sample_load_snapshot():
    if False:
        while True:
            i = 10
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.LoadSnapshotRequest()
    operation = client.load_snapshot(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)