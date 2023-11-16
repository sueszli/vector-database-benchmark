# ruff: isort: skip_file

# start_define_graph
from dagster import graph, op, ConfigurableResource


class Server(ConfigurableResource):
    def ping_server(self):
        ...


@op
def interact_with_server(server: Server):
    server.ping_server()


@graph
def do_stuff():
    interact_with_server()


# end_define_graph

# start_define_jobs
from dagster import ResourceDefinition

prod_server = ResourceDefinition.mock_resource()
local_server = ResourceDefinition.mock_resource()

prod_job = do_stuff.to_job(resource_defs={"server": prod_server}, name="do_stuff_prod")
local_job = do_stuff.to_job(
    resource_defs={"server": local_server}, name="do_stuff_local"
)
# end_define_jobs
