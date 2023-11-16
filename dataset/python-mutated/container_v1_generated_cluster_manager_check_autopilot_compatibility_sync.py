from google.cloud import container_v1

def sample_check_autopilot_compatibility():
    if False:
        while True:
            i = 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.CheckAutopilotCompatibilityRequest()
    response = client.check_autopilot_compatibility(request=request)
    print(response)