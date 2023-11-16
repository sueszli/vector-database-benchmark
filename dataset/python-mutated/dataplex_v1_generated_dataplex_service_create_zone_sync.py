from google.cloud import dataplex_v1

def sample_create_zone():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataplexServiceClient()
    zone = dataplex_v1.Zone()
    zone.type_ = 'CURATED'
    zone.resource_spec.location_type = 'MULTI_REGION'
    request = dataplex_v1.CreateZoneRequest(parent='parent_value', zone_id='zone_id_value', zone=zone)
    operation = client.create_zone(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)