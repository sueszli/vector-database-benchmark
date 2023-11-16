from google.cloud import dataplex_v1

def sample_update_zone():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    zone = dataplex_v1.Zone()
    zone.type_ = 'CURATED'
    zone.resource_spec.location_type = 'MULTI_REGION'
    request = dataplex_v1.UpdateZoneRequest(zone=zone)
    operation = client.update_zone(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)