from dagster import graph, op, ConfigurableResource

class Server(ConfigurableResource):

    def ping_server(self):
        if False:
            print('Hello World!')
        ...

@op
def interact_with_server(server: Server):
    if False:
        return 10
    server.ping_server()

@graph
def do_stuff():
    if False:
        for i in range(10):
            print('nop')
    interact_with_server()
from dagster import ResourceDefinition
prod_server = ResourceDefinition.mock_resource()
local_server = ResourceDefinition.mock_resource()
prod_job = do_stuff.to_job(resource_defs={'server': prod_server}, name='do_stuff_prod')
local_job = do_stuff.to_job(resource_defs={'server': local_server}, name='do_stuff_local')