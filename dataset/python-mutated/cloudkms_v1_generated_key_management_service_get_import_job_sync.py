from google.cloud import kms_v1

def sample_get_import_job():
    if False:
        return 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.GetImportJobRequest(name='name_value')
    response = client.get_import_job(request=request)
    print(response)