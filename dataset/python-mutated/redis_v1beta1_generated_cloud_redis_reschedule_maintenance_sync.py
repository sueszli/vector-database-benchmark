from google.cloud import redis_v1beta1

def sample_reschedule_maintenance():
    if False:
        for i in range(10):
            print('nop')
    client = redis_v1beta1.CloudRedisClient()
    request = redis_v1beta1.RescheduleMaintenanceRequest(name='name_value', reschedule_type='SPECIFIC_TIME')
    operation = client.reschedule_maintenance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)