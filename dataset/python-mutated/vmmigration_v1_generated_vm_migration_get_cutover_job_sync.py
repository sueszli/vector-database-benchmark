from google.cloud import vmmigration_v1

def sample_get_cutover_job():
    if False:
        while True:
            i = 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetCutoverJobRequest(name='name_value')
    response = client.get_cutover_job(request=request)
    print(response)