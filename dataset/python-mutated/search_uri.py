from google.cloud import webrisk_v1
from google.cloud.webrisk_v1 import SearchUrisResponse

def search_uri(uri: str, threat_type: webrisk_v1.ThreatType.MALWARE) -> SearchUrisResponse:
    if False:
        i = 10
        return i + 15
    'Checks whether a URI is on a given threatList.\n\n    Multiple threatLists may be searched in a single query. The response will list all\n    requested threatLists the URI was found to match. If the URI is not\n    found on any of the requested ThreatList an empty response will be returned.\n\n    Args:\n        uri: The URI to be checked for matches\n            Example: "http://testsafebrowsing.appspot.com/s/malware.html"\n        threat_type: The ThreatLists to search in. Multiple ThreatLists may be specified.\n            Example: threat_type = webrisk_v1.ThreatType.MALWARE\n\n    Returns:\n        SearchUrisResponse that contains a threat_type if the URI is present in the threatList.\n    '
    webrisk_client = webrisk_v1.WebRiskServiceClient()
    request = webrisk_v1.SearchUrisRequest()
    request.threat_types = [threat_type]
    request.uri = uri
    response = webrisk_client.search_uris(request)
    if response.threat.threat_types:
        print(f'The URI has the following threat: {response}')
    else:
        print('The URL is safe!')
    return response