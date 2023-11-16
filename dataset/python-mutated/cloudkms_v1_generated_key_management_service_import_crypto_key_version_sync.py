from google.cloud import kms_v1

def sample_import_crypto_key_version():
    if False:
        i = 10
        return i + 15
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.ImportCryptoKeyVersionRequest(rsa_aes_wrapped_key=b'rsa_aes_wrapped_key_blob', parent='parent_value', algorithm='EXTERNAL_SYMMETRIC_ENCRYPTION', import_job='import_job_value')
    response = client.import_crypto_key_version(request=request)
    print(response)