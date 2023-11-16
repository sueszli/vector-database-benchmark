from google.cloud import webrisk_v1beta1

def sample_search_hashes():
    if False:
        return 10
    client = webrisk_v1beta1.WebRiskServiceV1Beta1Client()
    request = webrisk_v1beta1.SearchHashesRequest(threat_types=['UNWANTED_SOFTWARE'])
    response = client.search_hashes(request=request)
    print(response)