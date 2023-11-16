from google.cloud import webrisk_v1

def sample_compute_threat_list_diff():
    if False:
        print('Hello World!')
    client = webrisk_v1.WebRiskServiceClient()
    request = webrisk_v1.ComputeThreatListDiffRequest(threat_type='SOCIAL_ENGINEERING_EXTENDED_COVERAGE')
    response = client.compute_threat_list_diff(request=request)
    print(response)