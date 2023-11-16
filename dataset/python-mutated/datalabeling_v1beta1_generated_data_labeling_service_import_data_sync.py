from google.cloud import datalabeling_v1beta1

def sample_import_data():
    if False:
        for i in range(10):
            print('nop')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.ImportDataRequest(name='name_value')
    operation = client.import_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)