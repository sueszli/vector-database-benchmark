from __future__ import annotations
import sys
from typing import Any
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

def wait_for_extended_operation(operation: ExtendedOperation, verbose_name: str='operation', timeout: int=300) -> Any:
    if False:
        i = 10
        return i + 15
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

def create_firewall_rule_for_windows_activation_host(project_id: str, firewall_rule_name: str, network: str='global/networks/default') -> compute_v1.Firewall:
    if False:
        i = 10
        return i + 15
    '\n    Creates an egress firewall rule with the highest priority for host\n    kms.windows.googlecloud.com (35.190.247.13) for Windows activation.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        firewall_rule_name: name of the rule that is created.\n        network: name of the network the rule will be applied to. Available name formats:\n            * https://www.googleapis.com/compute/v1/projects/{project_id}/global/networks/{network}\n            * projects/{project_id}/global/networks/{network}\n            * global/networks/{network}\n\n    Returns:\n        A Firewall object.\n    '
    firewall_rule = compute_v1.Firewall()
    firewall_rule.name = firewall_rule_name
    firewall_rule.network = network
    allowed = compute_v1.Allowed()
    allowed.ports = ['1688']
    allowed.I_p_protocol = 'tcp'
    firewall_rule.allowed = [allowed]
    firewall_rule.destination_ranges = ['35.190.247.13/32']
    firewall_rule.direction = compute_v1.Firewall.Direction.EGRESS.name
    firewall_rule.priority = 0
    firewall_client = compute_v1.FirewallsClient()
    operation = firewall_client.insert(project=project_id, firewall_resource=firewall_rule)
    wait_for_extended_operation(operation, 'windows KSM firewall rule creation')
    return firewall_client.get(project=project_id, firewall=firewall_rule_name)