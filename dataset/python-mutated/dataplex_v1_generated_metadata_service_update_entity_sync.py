from google.cloud import dataplex_v1

def sample_update_entity():
    if False:
        print('Hello World!')
    client = dataplex_v1.MetadataServiceClient()
    entity = dataplex_v1.Entity()
    entity.id = 'id_value'
    entity.type_ = 'FILESET'
    entity.asset = 'asset_value'
    entity.data_path = 'data_path_value'
    entity.system = 'BIGQUERY'
    entity.format_.mime_type = 'mime_type_value'
    entity.schema.user_managed = True
    request = dataplex_v1.UpdateEntityRequest(entity=entity)
    response = client.update_entity(request=request)
    print(response)