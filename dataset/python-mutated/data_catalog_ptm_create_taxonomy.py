from google.cloud import datacatalog_v1

def create_taxonomy(project_id: str='your-project-id', location_id: str='us', display_name: str='example-taxonomy'):
    if False:
        while True:
            i = 10
    client = datacatalog_v1.PolicyTagManagerClient()
    parent = datacatalog_v1.PolicyTagManagerClient.common_location_path(project_id, location_id)
    taxonomy = datacatalog_v1.Taxonomy()
    taxonomy.display_name = display_name
    taxonomy.description = 'This Taxonomy represents ...'
    taxonomy = client.create_taxonomy(parent=parent, taxonomy=taxonomy)
    print(f'Created taxonomy {taxonomy.name}')