from google.cloud import webrisk_v1

def sample_submit_uri():
    if False:
        return 10
    client = webrisk_v1.WebRiskServiceClient()
    submission = webrisk_v1.Submission()
    submission.uri = 'uri_value'
    request = webrisk_v1.SubmitUriRequest(parent='parent_value', submission=submission)
    operation = client.submit_uri(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)