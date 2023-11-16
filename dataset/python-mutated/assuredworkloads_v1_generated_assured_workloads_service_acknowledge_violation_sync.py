from google.cloud import assuredworkloads_v1

def sample_acknowledge_violation():
    if False:
        i = 10
        return i + 15
    client = assuredworkloads_v1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1.AcknowledgeViolationRequest(name='name_value', comment='comment_value')
    response = client.acknowledge_violation(request=request)
    print(response)