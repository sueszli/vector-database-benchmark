from google.cloud import compute_v1

def sample_set_usage_export_bucket():
    if False:
        while True:
            i = 10
    client = compute_v1.ProjectsClient()
    request = compute_v1.SetUsageExportBucketProjectRequest(project='project_value')
    response = client.set_usage_export_bucket(request=request)
    print(response)