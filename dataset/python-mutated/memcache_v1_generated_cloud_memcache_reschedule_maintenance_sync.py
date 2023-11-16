from google.cloud import memcache_v1

def sample_reschedule_maintenance():
    if False:
        print('Hello World!')
    client = memcache_v1.CloudMemcacheClient()
    request = memcache_v1.RescheduleMaintenanceRequest(instance='instance_value', reschedule_type='SPECIFIC_TIME')
    operation = client.reschedule_maintenance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)