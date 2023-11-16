from google.cloud import redis_v1

def sample_reschedule_maintenance():
    if False:
        while True:
            i = 10
    client = redis_v1.CloudRedisClient()
    request = redis_v1.RescheduleMaintenanceRequest(name='name_value', reschedule_type='SPECIFIC_TIME')
    operation = client.reschedule_maintenance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)