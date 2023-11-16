"""
A route is a rule that specifies how certain packets should be handled by the
virtual network. Routes are associated with virtual machine instances by tag,
and the set of routes for a particular VM is called its routing table.
For each packet leaving a virtual machine, the system searches that machine's
routing table for a single best matching route.

.. versionadded:: 2018.3.0

This module will create a route to send traffic destined to the Internet
through your gateway instance.

:codeauthor: `Pratik Bandarkar <pratik.bandarkar@gmail.com>`
:maturity:   new
:depends:    google-api-python-client
:platform:   Linux

"""
import logging
try:
    import googleapiclient.discovery
    import oauth2client.service_account
    HAS_LIB = True
except ImportError:
    HAS_LIB = False
log = logging.getLogger(__name__)
__virtualname__ = 'gcp'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Check for googleapiclient api\n    '
    if HAS_LIB is False:
        return (False, "Required dependencies 'googleapiclient' and/or 'oauth2client' were not found.")
    return __virtualname__

def _get_network(project_id, network_name, service):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fetch network selfLink from network name.\n    '
    return service.networks().get(project=project_id, network=network_name).execute()

def _get_instance(project_id, instance_zone, name, service):
    if False:
        return 10
    '\n    Get instance details\n    '
    return service.instances().get(project=project_id, zone=instance_zone, instance=name).execute()

def route_create(credential_file=None, project_id=None, name=None, dest_range=None, next_hop_instance=None, instance_zone=None, tags=None, network=None, priority=None):
    if False:
        return 10
    '\n    Create a route to send traffic destined to the Internet through your\n    gateway instance\n\n    credential_file : string\n        File location of application default credential. For more information,\n        refer: https://developers.google.com/identity/protocols/application-default-credentials\n    project_id : string\n        Project ID where instance and network resides.\n    name : string\n        name of the route to create\n    next_hop_instance : string\n        the name of an instance that should handle traffic matching this route.\n    instance_zone : string\n        zone where instance("next_hop_instance") resides\n    network : string\n        Specifies the network to which the route will be applied.\n    dest_range : string\n        The destination range of outgoing packets that the route will apply to.\n    tags : list\n        (optional) Identifies the set of instances that this route will apply to.\n    priority : int\n        (optional) Specifies the priority of this route relative to other routes.\n        default=1000\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'salt-master.novalocal\' gcp.route_create\n            credential_file=/root/secret_key.json\n            project_id=cp100-170315\n            name=derby-db-route1\n            next_hop_instance=instance-1\n            instance_zone=us-central1-a\n            network=default\n            dest_range=0.0.0.0/0\n            tags=[\'no-ip\']\n            priority=700\n\n    In above example, the instances which are having tag "no-ip" will route the\n    packet to instance "instance-1"(if packet is intended to other network)\n    '
    credentials = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(credential_file)
    service = googleapiclient.discovery.build('compute', 'v1', credentials=credentials)
    routes = service.routes()
    routes_config = {'name': str(name), 'network': _get_network(project_id, str(network), service=service)['selfLink'], 'destRange': str(dest_range), 'nextHopInstance': _get_instance(project_id, instance_zone, next_hop_instance, service=service)['selfLink'], 'tags': tags, 'priority': priority}
    route_create_request = routes.insert(project=project_id, body=routes_config)
    return route_create_request.execute()