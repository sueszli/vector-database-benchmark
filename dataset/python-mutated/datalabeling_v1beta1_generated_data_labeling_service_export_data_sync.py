from google.cloud import datalabeling_v1beta1

def sample_export_data():
    if False:
        for i in range(10):
            print('nop')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.ExportDataRequest(name='name_value', annotated_dataset='annotated_dataset_value')
    operation = client.export_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)