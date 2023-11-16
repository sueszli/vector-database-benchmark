from google.cloud.orchestration.airflow import service_v1

def sample_save_snapshot():
    if False:
        i = 10
        return i + 15
    client = service_v1.EnvironmentsClient()
    request = service_v1.SaveSnapshotRequest()
    operation = client.save_snapshot(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)