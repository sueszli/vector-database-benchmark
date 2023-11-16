from __future__ import annotations
import sys
from typing import Any
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

def wait_for_extended_operation(operation: ExtendedOperation, verbose_name: str='operation', timeout: int=300) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Waits for the extended (long-running) operation to complete.\n\n    If the operation is successful, it will return its result.\n    If the operation ends with an error, an exception will be raised.\n    If there were any warnings during the execution of the operation\n    they will be printed to sys.stderr.\n\n    Args:\n        operation: a long-running operation you want to wait on.\n        verbose_name: (optional) a more verbose name of the operation,\n            used only during error and warning reporting.\n        timeout: how long (in seconds) to wait for operation to finish.\n            If None, wait indefinitely.\n\n    Returns:\n        Whatever the operation.result() returns.\n\n    Raises:\n        This method will raise the exception received from `operation.exception()`\n        or RuntimeError if there is no exception set, but there is an `error_code`\n        set for the `operation`.\n\n        In case of an operation taking longer than `timeout` seconds to complete,\n        a `concurrent.futures.TimeoutError` will be raised.\n    '
    result = operation.result(timeout=timeout)
    if operation.error_code:
        print(f'Error during {verbose_name}: [Code: {operation.error_code}]: {operation.error_message}', file=sys.stderr, flush=True)
        print(f'Operation ID: {operation.name}', file=sys.stderr, flush=True)
        raise operation.exception() or RuntimeError(operation.error_message)
    if operation.warnings:
        print(f'Warnings during {verbose_name}:\n', file=sys.stderr, flush=True)
        for warning in operation.warnings:
            print(f' - {warning.code}: {warning.message}', file=sys.stderr, flush=True)
    return result

def create_route(project_id: str, network: str, route_name: str, destination_range: str, *, next_hop_gateway: str | None=None, next_hop_ip: str | None=None, next_hop_instance: str | None=None, next_hop_vpn_tunnel: str | None=None, next_hop_ilb: str | None=None) -> compute_v1.Route:
    if False:
        i = 10
        return i + 15
    '\n    Create a new route in selected network by providing a destination and next hop name.\n\n    Note: The set of {next_hop_gateway, next_hop_ip, next_hop_instance, next_hop_vpn_tunnel,\n        next_hop_ilb} is exclusive, you and only specify one of those parameters.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        network: name of the network the route will be created in. Available name formats:\n            * https://www.googleapis.com/compute/v1/projects/{project_id}/global/networks/{network}\n            * projects/{project_id}/global/networks/{network}\n            * global/networks/{network}\n        route_name: name of the new route.\n        destination_range: range of destination IPs this route should be applied to. E.g. 10.0.0.0/16.\n        next_hop_gateway: name of the gateway the traffic should be directed to.\n        next_hop_ip: IP address the traffic should be directed to.\n        next_hop_instance: name of the instance the traffic should be directed to. Name format:\n            "projects/{project}/zones/{zone}/instances/{instance_name}"\n        next_hop_vpn_tunnel: name of the VPN tunnel the traffic should be directed to. Name format:\n            "projects/{project}/regions/{region}/vpnTunnels/{vpn_tunnel_name}"\n        next_hop_ilb: name of a forwarding rule of the Internal Load Balancer the traffic\n            should be directed to. Name format:\n            "projects/{project}/regions/{region}/forwardingRules/{forwarding_rule_region}"\n\n    Returns:\n        A new compute_v1.Route object.\n    '
    excl_args = {next_hop_instance, next_hop_ilb, next_hop_vpn_tunnel, next_hop_gateway, next_hop_ip}
    args_set = sum((1 if arg is not None else 0 for arg in excl_args))
    if args_set != 1:
        raise RuntimeError('You must specify exactly one next_hop_* parameter.')
    route = compute_v1.Route()
    route.name = route_name
    route.network = network
    route.dest_range = destination_range
    if next_hop_gateway:
        route.next_hop_gateway = next_hop_gateway
    elif next_hop_ip:
        route.next_hop_ip = next_hop_ip
    elif next_hop_instance:
        route.next_hop_instance = next_hop_instance
    elif next_hop_vpn_tunnel:
        route.next_hop_vpn_tunnel = next_hop_vpn_tunnel
    elif next_hop_ilb:
        route.next_hop_ilb = next_hop_ilb
    route_client = compute_v1.RoutesClient()
    operation = route_client.insert(project=project_id, route_resource=route)
    wait_for_extended_operation(operation, 'route creation')
    return route_client.get(project=project_id, route=route_name)

def create_route_to_windows_activation_host(project_id: str, network: str, route_name: str) -> compute_v1.Route:
    if False:
        while True:
            i = 10
    '\n    If you have Windows instances without external IP addresses,\n    you must also enable Private Google Access so that instances\n    with only internal IP addresses can send traffic to the external\n    IP address for kms.windows.googlecloud.com.\n    More infromation: https://cloud.google.com/vpc/docs/configure-private-google-access#enabling\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        network: name of the network the route will be created in. Available name formats:\n            * https://www.googleapis.com/compute/v1/projects/{project_id}/global/networks/{network}\n            * projects/{project_id}/global/networks/{network}\n            * global/networks/{network}\n        route_name: name of the new route.\n\n    Returns:\n        A new compute_v1.Route object.\n    '
    return create_route(project_id=project_id, network=network, route_name=route_name, destination_range='35.190.247.13/32', next_hop_gateway=f'projects/{project_id}/global/gateways/default-internet-gateway')