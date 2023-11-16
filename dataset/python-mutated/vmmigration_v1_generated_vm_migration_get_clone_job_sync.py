from google.cloud import vmmigration_v1

def sample_get_clone_job():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetCloneJobRequest(name='name_value')
    response = client.get_clone_job(request=request)
    print(response)