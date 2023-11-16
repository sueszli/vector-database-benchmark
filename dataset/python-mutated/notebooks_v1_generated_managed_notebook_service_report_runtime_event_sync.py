from google.cloud import notebooks_v1

def sample_report_runtime_event():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.ReportRuntimeEventRequest(name='name_value', vm_id='vm_id_value')
    operation = client.report_runtime_event(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)