from google.cloud import container_v1beta1

def sample_check_autopilot_compatibility():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.CheckAutopilotCompatibilityRequest()
    response = client.check_autopilot_compatibility(request=request)
    print(response)