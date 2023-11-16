from google.cloud import webrisk_v1

def sample_search_hashes():
    if False:
        i = 10
        return i + 15
    client = webrisk_v1.WebRiskServiceClient()
    request = webrisk_v1.SearchHashesRequest(threat_types=['SOCIAL_ENGINEERING_EXTENDED_COVERAGE'])
    response = client.search_hashes(request=request)
    print(response)