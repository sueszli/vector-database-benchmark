from google.cloud import edgenetwork_v1

def sample_update_router():
    if False:
        print('Hello World!')
    client = edgenetwork_v1.EdgeNetworkClient()
    router = edgenetwork_v1.Router()
    router.name = 'name_value'
    router.network = 'network_value'
    request = edgenetwork_v1.UpdateRouterRequest(router=router)
    operation = client.update_router(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)