from google.cloud import memcache_v1beta2

def sample_reschedule_maintenance():
    if False:
        i = 10
        return i + 15
    client = memcache_v1beta2.CloudMemcacheClient()
    request = memcache_v1beta2.RescheduleMaintenanceRequest(instance='instance_value', reschedule_type='SPECIFIC_TIME')
    operation = client.reschedule_maintenance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)