import argparse

def update_saved_query(saved_query_name, description):
    if False:
        return 10
    from google.cloud import asset_v1
    from google.protobuf import field_mask_pb2
    client = asset_v1.AssetServiceClient()
    saved_query = asset_v1.SavedQuery()
    saved_query.name = saved_query_name
    saved_query.description = description
    update_mask = field_mask_pb2.FieldMask()
    update_mask.paths.append('description')
    response = client.update_saved_query(request={'saved_query': saved_query, 'update_mask': update_mask})
    print(f'updated_saved_query: {response}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('saved_query_name', help='SavedQuery Name you want to update')
    parser.add_argument('description', help='The description you want to update with')
    args = parser.parse_args()
    update_saved_query(args.saved_query_name, args.description)