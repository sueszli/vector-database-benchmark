from google.cloud import kms_v1

def sample_create_import_job():
    if False:
        return 10
    client = kms_v1.KeyManagementServiceClient()
    import_job = kms_v1.ImportJob()
    import_job.import_method = 'RSA_OAEP_4096_SHA256'
    import_job.protection_level = 'EXTERNAL_VPC'
    request = kms_v1.CreateImportJobRequest(parent='parent_value', import_job_id='import_job_id_value', import_job=import_job)
    response = client.create_import_job(request=request)
    print(response)