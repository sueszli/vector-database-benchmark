from google.cloud import webrisk_v1

def search_hashes(hash_prefix: bytes, threat_type: webrisk_v1.ThreatType) -> list:
    if False:
        for i in range(10):
            print('nop')
    'Gets the full hashes that match the requested hash prefix.\n\n    This is used after a hash prefix is looked up in a threatList and there is a match.\n    The client side threatList only holds partial hashes so the client must query this method\n    to determine if there is a full hash match of a threat.\n\n    Args:\n        hash_prefix: A hash prefix, consisting of the most significant 4-32 bytes of a SHA256 hash.\n            For JSON requests, this field is base64-encoded. Note that if this parameter is provided\n            by a URI, it must be encoded using the web safe base64 variant (RFC 4648).\n            Example:\n                uri = "http://example.com"\n                sha256 = sha256()\n                sha256.update(base64.urlsafe_b64encode(bytes(uri, "utf-8")))\n                hex_string = sha256.digest()\n\n        threat_type: The ThreatLists to search in. Multiple ThreatLists may be specified.\n            For the list on threat types, see:\n            https://cloud.google.com/web-risk/docs/reference/rpc/google.cloud.webrisk.v1#threattype\n            threat_type = [webrisk_v1.ThreatType.MALWARE, webrisk_v1.ThreatType.SOCIAL_ENGINEERING]\n\n    Returns:\n        A hash list that contain all hashes that matches the given hash prefix.\n    '
    webrisk_client = webrisk_v1.WebRiskServiceClient()
    request = webrisk_v1.SearchHashesRequest()
    request.hash_prefix = hash_prefix
    request.threat_types = [threat_type]
    response = webrisk_client.search_hashes(request)
    hash_list = []
    for threat_hash in response.threats:
        hash_list.append(threat_hash.hash)
    return hash_list