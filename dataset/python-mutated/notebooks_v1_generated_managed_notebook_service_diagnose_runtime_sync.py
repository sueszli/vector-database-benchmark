from google.cloud import notebooks_v1

def sample_diagnose_runtime():
    if False:
        return 10
    client = notebooks_v1.ManagedNotebookServiceClient()
    diagnostic_config = notebooks_v1.DiagnosticConfig()
    diagnostic_config.gcs_bucket = 'gcs_bucket_value'
    request = notebooks_v1.DiagnoseRuntimeRequest(name='name_value', diagnostic_config=diagnostic_config)
    operation = client.diagnose_runtime(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)