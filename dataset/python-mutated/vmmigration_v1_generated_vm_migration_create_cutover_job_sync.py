from google.cloud import vmmigration_v1

def sample_create_cutover_job():
    if False:
        i = 10
        return i + 15
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CreateCutoverJobRequest(parent='parent_value', cutover_job_id='cutover_job_id_value')
    operation = client.create_cutover_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)