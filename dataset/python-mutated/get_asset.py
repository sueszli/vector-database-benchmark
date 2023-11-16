"""Google Cloud Live Stream sample for getting an asset.
Example usage:
    python get_asset.py --project_id <project-id> --location <location> --asset_id <asset-id>
"""
import argparse
from google.cloud.video import live_stream_v1
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient

def get_asset(project_id: str, location: str, asset_id: str) -> live_stream_v1.types.Asset:
    if False:
        for i in range(10):
            print('nop')
    'Gets an asset.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the asset.\n        asset_id: The user-defined asset ID.'
    client = LivestreamServiceClient()
    name = f'projects/{project_id}/locations/{location}/assets/{asset_id}'
    response = client.get_asset(name=name)
    print(f'Asset: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the asset.', required=True)
    parser.add_argument('--asset_id', help='The user-defined asset ID.', required=True)
    args = parser.parse_args()
    get_asset(args.project_id, args.location, args.asset_id)