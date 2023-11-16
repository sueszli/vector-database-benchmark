from google.cloud import webrisk_v1beta1

def sample_compute_threat_list_diff():
    if False:
        return 10
    client = webrisk_v1beta1.WebRiskServiceV1Beta1Client()
    request = webrisk_v1beta1.ComputeThreatListDiffRequest(threat_type='UNWANTED_SOFTWARE')
    response = client.compute_threat_list_diff(request=request)
    print(response)