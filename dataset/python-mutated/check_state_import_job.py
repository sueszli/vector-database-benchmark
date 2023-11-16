from google.cloud import kms

def check_state_import_job(project_id: str, location_id: str, key_ring_id: str, import_job_id: str) -> None:
    if False:
        print('Hello World!')
    "\n    Check the state of an import job in Cloud KMS.\n\n    Args:\n        project_id (string): Google Cloud project ID (e.g. 'my-project').\n        location_id (string): Cloud KMS location (e.g. 'us-east1').\n        key_ring_id (string): ID of the Cloud KMS key ring (e.g. 'my-key-ring').\n        import_job_id (string): ID of the import job (e.g. 'my-import-job').\n    "
    client = kms.KeyManagementServiceClient()
    import_job_name = client.import_job_path(project_id, location_id, key_ring_id, import_job_id)
    import_job = client.get_import_job(name=import_job_name)
    print(f'Current state of import job {import_job.name}: {import_job.state}')