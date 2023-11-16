import argparse

def analyze_org_policies(organization_id, constraint):
    if False:
        for i in range(10):
            print('nop')
    from google.cloud import asset_v1
    client = asset_v1.AssetServiceClient()
    scope = f'organizations/{organization_id}'
    response = client.analyze_org_policies(request={'scope': scope, 'constraint': constraint})
    print(f'Analysis completed successfully: {response}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('organization_id', help='Your Google Cloud Organization ID')
    parser.add_argument('constraint', help='Constraint you want to analyze')
    args = parser.parse_args()
    analyze_org_policies(args.organization_id, args.constraint)