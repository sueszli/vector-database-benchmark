from google.cloud import webrisk_v1

def sample_create_submission():
    if False:
        return 10
    client = webrisk_v1.WebRiskServiceClient()
    submission = webrisk_v1.Submission()
    submission.uri = 'uri_value'
    request = webrisk_v1.CreateSubmissionRequest(parent='parent_value', submission=submission)
    response = client.create_submission(request=request)
    print(response)