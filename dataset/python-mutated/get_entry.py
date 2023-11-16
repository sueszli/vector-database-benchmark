def sample_get_entry(project_id: str, location_id: str, entry_group_id: str, entry_id: str):
    if False:
        return 10
    from google.cloud import datacatalog_v1beta1
    '\n    Get Entry\n\n    Args:\n      project_id (str): Your Google Cloud project ID\n      location_id (str): Google Cloud region, e.g. us-central1\n      entry_group_id (str): ID of the Entry Group, e.g. @bigquery, @pubsub, my_entry_group\n      entry_id (str): ID of the Entry\n    '
    client = datacatalog_v1beta1.DataCatalogClient()
    name = client.entry_path(project_id, location_id, entry_group_id, entry_id)
    entry = client.get_entry(request={'name': name})
    print(f'Entry name: {entry.name}')
    print(f'Entry type: {datacatalog_v1beta1.EntryType(entry.type_).name}')
    print(f'Linked resource: {entry.linked_resource}')
    return entry

def main():
    if False:
        i = 10
        return i + 15
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type_=str, default='[Google Cloud Project ID]')
    parser.add_argument('--location_id', type_=str, default='[Google Cloud Location ID]')
    parser.add_argument('--entry_group_id', type_=str, default='[Entry Group ID]')
    parser.add_argument('--entry_id', type_=str, default='[Entry ID]')
    args = parser.parse_args()
    sample_get_entry(args.project_id, args.location_id, args.entry_group_id, args.entry_id)
if __name__ == '__main__':
    main()