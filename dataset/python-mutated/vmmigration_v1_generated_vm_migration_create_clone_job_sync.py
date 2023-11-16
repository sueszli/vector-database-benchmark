from google.cloud import vmmigration_v1

def sample_create_clone_job():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CreateCloneJobRequest(parent='parent_value', clone_job_id='clone_job_id_value')
    operation = client.create_clone_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)