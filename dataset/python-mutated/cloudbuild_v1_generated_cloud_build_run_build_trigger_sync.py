from google.cloud.devtools import cloudbuild_v1

def sample_run_build_trigger():
    if False:
        for i in range(10):
            print('nop')
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.RunBuildTriggerRequest(project_id='project_id_value', trigger_id='trigger_id_value')
    operation = client.run_build_trigger(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)