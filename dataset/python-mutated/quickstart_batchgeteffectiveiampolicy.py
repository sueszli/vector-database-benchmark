import argparse

def batch_get_effective_iam_policies(resource_names, scope):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    response = client.batch_get_effective_iam_policies(request={'scope': scope, 'names': resource_names})
    print(response)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('resource_names', help='Your specified accessible scope, such as a project, folder or organization')
    parser.add_argument('scope', help='Your specified list of resource names')
    args = parser.parse_args()
    batch_get_effective_iam_policies(args.resource_names, args.scope)