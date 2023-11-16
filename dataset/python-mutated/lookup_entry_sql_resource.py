def sample_lookup_entry(sql_name: str):
    if False:
        i = 10
        return i + 15
    from google.cloud import datacatalog_v1beta1
    '\n    Lookup Entry using SQL resource\n\n    Args:\n      sql_name (str): The SQL name of the Google Cloud Platform resource the Data Catalog\n      entry represents.\n      Examples:\n      bigquery.table.`bigquery-public-data`.new_york_taxi_trips.taxi_zone_geom\n      pubsub.topic.`pubsub-public-data`.`taxirides-realtime`\n    '
    client = datacatalog_v1beta1.DataCatalogClient()
    entry = client.lookup_entry(request={'sql_resource': sql_name})
    print(f'Entry name: {entry.name}')
    print(f'Entry type: {datacatalog_v1beta1.EntryType(entry.type_).name}')
    print(f'Linked resource: {entry.linked_resource}')
    return entry

def main():
    if False:
        print('Hello World!')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sql_name', type=str, default='[SQL Resource Name]')
    args = parser.parse_args()
    sample_lookup_entry(args.sql_name)
if __name__ == '__main__':
    main()