from google.cloud.orchestration.airflow import service_v1beta1

def sample_save_snapshot():
    if False:
        print('Hello World!')
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.SaveSnapshotRequest()
    operation = client.save_snapshot(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)