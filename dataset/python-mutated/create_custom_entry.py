def create_custom_entry(override_values):
    if False:
        print('Hello World!')
    'Creates a custom entry within an entry group.'
    from google.cloud import datacatalog_v1
    project_id = 'my-project'
    entry_group_id = 'my_new_entry_group_id'
    entry_id = 'my_new_entry_id'
    location = 'us-central1'
    project_id = override_values.get('project_id', project_id)
    entry_id = override_values.get('entry_id', entry_id)
    entry_group_id = override_values.get('entry_group_id', entry_group_id)
    datacatalog = datacatalog_v1.DataCatalogClient()
    entry_group_obj = datacatalog_v1.types.EntryGroup()
    entry_group_obj.display_name = 'My awesome Entry Group'
    entry_group_obj.description = 'This Entry Group represents an external system'
    entry_group = datacatalog.create_entry_group(parent=datacatalog_v1.DataCatalogClient.common_location_path(project_id, location), entry_group_id=entry_group_id, entry_group=entry_group_obj)
    entry_group_name = entry_group.name
    print('Created entry group: {}'.format(entry_group_name))
    entry = datacatalog_v1.types.Entry()
    entry.user_specified_system = 'onprem_data_system'
    entry.user_specified_type = 'onprem_data_asset'
    entry.display_name = 'My awesome data asset'
    entry.description = 'This data asset is managed by an external system.'
    entry.linked_resource = '//my-onprem-server.com/dataAssets/my-awesome-data-asset'
    entry.schema.columns.append(datacatalog_v1.types.ColumnSchema(column='first_column', type_='STRING', description='This columns consists of ....', mode=None))
    entry.schema.columns.append(datacatalog_v1.types.ColumnSchema(column='second_column', type_='DOUBLE', description='This columns consists of ....', mode=None))
    entry = datacatalog.create_entry(parent=entry_group_name, entry_id=entry_id, entry=entry)
    print('Created entry: {}'.format(entry.name))