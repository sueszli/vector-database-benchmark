from google.cloud import webrisk_v1
from google.cloud.webrisk_v1 import ComputeThreatListDiffResponse

def compute_threatlist_diff(threat_type: webrisk_v1.ThreatType, version_token: bytes, max_diff_entries: int, max_database_entries: int, compression_type: webrisk_v1.CompressionType) -> ComputeThreatListDiffResponse:
    if False:
        for i in range(10):
            print('nop')
    'Gets the most recent threat list diffs.\n\n    These diffs should be applied to a local database of hashes to keep it up-to-date.\n    If the local database is empty or excessively out-of-date,\n    a complete snapshot of the database will be returned. This Method only updates a\n    single ThreatList at a time. To update multiple ThreatList databases, this method needs to be\n    called once for each list.\n\n    Args:\n        threat_type: The threat list to update. Only a single ThreatType should be specified per request.\n            threat_type = webrisk_v1.ThreatType.MALWARE\n\n        version_token: The current version token of the client for the requested list. If the\n            client does not have a version token (this is the first time calling ComputeThreatListDiff),\n            this may be left empty and a full database snapshot will be returned.\n\n        max_diff_entries: The maximum size in number of entries. The diff will not contain more entries\n            than this value. This should be a power of 2 between 2**10 and 2**20.\n            If zero, no diff size limit is set.\n            max_diff_entries = 1024\n\n        max_database_entries: Sets the maximum number of entries that the client is willing to have in the local database.\n            This should be a power of 2 between 2**10 and 2**20. If zero, no database size limit is set.\n            max_database_entries = 1024\n\n        compression_type: The compression type supported by the client.\n            compression_type = webrisk_v1.CompressionType.RAW\n\n    Returns:\n        The response which contains the diff between local and remote threat lists. In addition to the threat list,\n        the response also contains the version token and the recommended time for next diff.\n    '
    webrisk_client = webrisk_v1.WebRiskServiceClient()
    constraints = webrisk_v1.ComputeThreatListDiffRequest.Constraints()
    constraints.max_diff_entries = max_diff_entries
    constraints.max_database_entries = max_database_entries
    constraints.supported_compressions = [compression_type]
    request = webrisk_v1.ComputeThreatListDiffRequest()
    request.threat_type = threat_type
    request.version_token = version_token
    request.constraints = constraints
    response = webrisk_client.compute_threat_list_diff(request)
    print(response.response_type)
    print(response.new_version_token)
    print(response.recommended_next_diff)
    return response