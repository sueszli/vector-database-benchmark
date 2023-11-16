from google.cloud import notebooks_v1

def sample_diagnose_instance():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    diagnostic_config = notebooks_v1.DiagnosticConfig()
    diagnostic_config.gcs_bucket = 'gcs_bucket_value'
    request = notebooks_v1.DiagnoseInstanceRequest(name='name_value', diagnostic_config=diagnostic_config)
    operation = client.diagnose_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)