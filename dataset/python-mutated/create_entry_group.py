def create_entry_group(project_id, entry_group_id):
    if False:
        for i in range(10):
            print('nop')
    from google.cloud import datacatalog_v1beta1
    client = datacatalog_v1beta1.DataCatalogClient()
    location_id = 'us-central1'
    parent = f'projects/{project_id}/locations/{location_id}'
    entry_group = datacatalog_v1beta1.EntryGroup()
    entry_group.display_name = 'My Entry Group'
    entry_group.description = 'This Entry Group consists of ...'
    entry_group = client.create_entry_group(request={'parent': parent, 'entry_group_id': entry_group_id, 'entry_group': entry_group})
    print('Created entry group {}'.format(entry_group.name))