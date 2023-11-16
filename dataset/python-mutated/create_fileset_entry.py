def create_fileset_entry(client, entry_group_name, entry_id):
    if False:
        for i in range(10):
            print('nop')
    from google.cloud import datacatalog_v1beta1
    entry = datacatalog_v1beta1.types.Entry()
    entry.display_name = 'My Fileset'
    entry.description = 'This Fileset consists of ...'
    entry.gcs_fileset_spec.file_patterns.append('gs://my_bucket/*')
    entry.type_ = datacatalog_v1beta1.EntryType.FILESET
    columns = []
    columns.append(datacatalog_v1beta1.types.ColumnSchema(column='first_name', description='First name', mode='REQUIRED', type_='STRING'))
    columns.append(datacatalog_v1beta1.types.ColumnSchema(column='last_name', description='Last name', mode='REQUIRED', type_='STRING'))
    subcolumns = []
    subcolumns.append(datacatalog_v1beta1.types.ColumnSchema(column='city', description='City', mode='NULLABLE', type_='STRING'))
    subcolumns.append(datacatalog_v1beta1.types.ColumnSchema(column='state', description='State', mode='NULLABLE', type_='STRING'))
    columns.append(datacatalog_v1beta1.types.ColumnSchema(column='addresses', description='Addresses', mode='REPEATED', subcolumns=subcolumns, type_='RECORD'))
    entry.schema.columns.extend(columns)
    entry = client.create_entry(request={'parent': entry_group_name, 'entry_id': entry_id, 'entry': entry})
    print('Created entry {}'.format(entry.name))