from google.cloud import kms

def create_import_job(project_id: str, location_id: str, key_ring_id: str, import_job_id: str) -> None:
    if False:
        i = 10
        return i + 15
    "\n    Create a new import job in Cloud KMS.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        import_job_id (string): ID of the import job (e.g. 'my-import-job').\n    "
    client = kms.KeyManagementServiceClient()
    key_ring_name = client.key_ring_path(project_id, location_id, key_ring_id)
    import_method = kms.ImportJob.ImportMethod.RSA_OAEP_3072_SHA1_AES_256
    protection_level = kms.ProtectionLevel.HSM
    import_job_params = {'import_method': import_method, 'protection_level': protection_level}
    import_job = client.create_import_job({'parent': key_ring_name, 'import_job_id': import_job_id, 'import_job': import_job_params})
    print(f'Created import job: {import_job.name}')