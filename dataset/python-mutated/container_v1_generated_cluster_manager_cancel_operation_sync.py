from google.cloud import container_v1

def sample_cancel_operation():
    if False:
        while True:
            i = 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.CancelOperationRequest()
    client.cancel_operation(request=request)