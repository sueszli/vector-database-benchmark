from google.cloud import appengine_admin_v1

def sample_repair_application():
    if False:
        i = 10
        return i + 15
    client = appengine_admin_v1.ApplicationsClient()
    request = appengine_admin_v1.RepairApplicationRequest()
    operation = client.repair_application(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)