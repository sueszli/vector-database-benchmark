from google.cloud import notebooks_v1beta1

def sample_report_instance_info():
    if False:
        print('Hello World!')
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.ReportInstanceInfoRequest(name='name_value', vm_id='vm_id_value')
    operation = client.report_instance_info(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)