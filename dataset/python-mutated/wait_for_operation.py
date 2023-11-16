from google.cloud import compute_v1

def wait_for_operation(operation: compute_v1.Operation, project_id: str) -> compute_v1.Operation:
    if False:
        for i in range(10):
            print('nop')
    '\n    This method waits for an operation to be completed. Calling this function\n    will block until the operation is finished.\n\n    Args:\n        operation: The Operation object representing the operation you want to\n            wait on.\n        project_id: project ID or project number of the Cloud project you want to use.\n\n    Returns:\n        Finished Operation object.\n    '
    kwargs = {'project': project_id, 'operation': operation.name}
    if operation.zone:
        client = compute_v1.ZoneOperationsClient()
        kwargs['zone'] = operation.zone.rsplit('/', maxsplit=1)[1]
    elif operation.region:
        client = compute_v1.RegionOperationsClient()
        kwargs['region'] = operation.region.rsplit('/', maxsplit=1)[1]
    else:
        client = compute_v1.GlobalOperationsClient()
    return client.wait(**kwargs)