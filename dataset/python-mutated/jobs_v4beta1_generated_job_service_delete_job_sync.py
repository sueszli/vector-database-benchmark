from google.cloud import talent_v4beta1

def sample_delete_job():
    if False:
        i = 10
        return i + 15
    client = talent_v4beta1.JobServiceClient()
    request = talent_v4beta1.DeleteJobRequest(name='name_value')
    client.delete_job(request=request)