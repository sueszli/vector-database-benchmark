from google.cloud import edgenetwork_v1

def sample_create_router():
    if False:
        i = 10
        return i + 15
    client = edgenetwork_v1.EdgeNetworkClient()
    router = edgenetwork_v1.Router()
    router.name = 'name_value'
    router.network = 'network_value'
    request = edgenetwork_v1.CreateRouterRequest(parent='parent_value', router_id='router_id_value', router=router)
    operation = client.create_router(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)