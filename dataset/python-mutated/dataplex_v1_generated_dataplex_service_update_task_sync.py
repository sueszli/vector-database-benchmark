from google.cloud import dataplex_v1

def sample_update_task():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataplexServiceClient()
    task = dataplex_v1.Task()
    task.spark.main_jar_file_uri = 'main_jar_file_uri_value'
    task.trigger_spec.schedule = 'schedule_value'
    task.trigger_spec.type_ = 'RECURRING'
    task.execution_spec.service_account = 'service_account_value'
    request = dataplex_v1.UpdateTaskRequest(task=task)
    operation = client.update_task(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)