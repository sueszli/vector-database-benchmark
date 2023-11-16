from google.cloud import dataplex_v1

def sample_create_entity():
    if False:
        return 10
    client = dataplex_v1.MetadataServiceClient()
    entity = dataplex_v1.Entity()
    entity.id = 'id_value'
    entity.type_ = 'FILESET'
    entity.asset = 'asset_value'
    entity.data_path = 'data_path_value'
    entity.system = 'BIGQUERY'
    entity.format_.mime_type = 'mime_type_value'
    entity.schema.user_managed = True
    request = dataplex_v1.CreateEntityRequest(parent='parent_value', entity=entity)
    response = client.create_entity(request=request)
    print(response)