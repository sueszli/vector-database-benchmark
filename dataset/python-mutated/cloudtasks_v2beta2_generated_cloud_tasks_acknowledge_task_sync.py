from google.cloud import tasks_v2beta2

def sample_acknowledge_task():
    if False:
        return 10
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.AcknowledgeTaskRequest(name='name_value')
    client.acknowledge_task(request=request)