from google.cloud import tasks_v2beta2

def sample_upload_queue_yaml():
    if False:
        while True:
            i = 10
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.UploadQueueYamlRequest(app_id='app_id_value')
    client.upload_queue_yaml(request=request)