def create_fileset(override_values):
    if False:
        while True:
            i = 10
    'Creates a fileset within an entry group.'
    from google.cloud import datacatalog_v1
    project_id = 'project_id'
    fileset_entry_group_id = 'entry_group_id'
    fileset_entry_id = 'entry_id'
    project_id = override_values.get('project_id', project_id)
    fileset_entry_group_id = override_values.get('fileset_entry_group_id', fileset_entry_group_id)
    fileset_entry_id = override_values.get('fileset_entry_id', fileset_entry_id)
    location = 'us-central1'
    datacatalog = datacatalog_v1.DataCatalogClient()
    entry_group_obj = datacatalog_v1.types.EntryGroup()
    entry_group_obj.display_name = 'My Fileset Entry Group'
    entry_group_obj.description = 'This Entry Group consists of ....'
    entry_group = datacatalog.create_entry_group(parent=datacatalog_v1.DataCatalogClient.common_location_path(project_id, location), entry_group_id=fileset_entry_group_id, entry_group=entry_group_obj)
    print(f'Created entry group: {entry_group.name}')
    entry = datacatalog_v1.types.Entry()
    entry.display_name = 'My Fileset'
    entry.description = 'This fileset consists of ....'
    entry.gcs_fileset_spec.file_patterns.append('gs://my_bucket/*.csv')
    entry.type_ = datacatalog_v1.EntryType.FILESET
    entry.schema.columns.append(datacatalog_v1.types.ColumnSchema(column='first_name', description='First name', mode='REQUIRED', type_='STRING'))
    entry.schema.columns.append(datacatalog_v1.types.ColumnSchema(column='last_name', description='Last name', mode='REQUIRED', type_='STRING'))
    addresses_column = datacatalog_v1.types.ColumnSchema(column='addresses', description='Addresses', mode='REPEATED', type_='RECORD')
    addresses_column.subcolumns.append(datacatalog_v1.types.ColumnSchema(column='city', description='City', mode='NULLABLE', type_='STRING'))
    addresses_column.subcolumns.append(datacatalog_v1.types.ColumnSchema(column='state', description='State', mode='NULLABLE', type_='STRING'))
    entry.schema.columns.append(addresses_column)
    entry = datacatalog.create_entry(parent=entry_group.name, entry_id=fileset_entry_id, entry=entry)
    print(f'Created fileset entry: {entry.name}')