"""Google Cloud Live Stream sample for deleting an asset.
Example usage:
    python delete_asset.py --project_id <project-id> --location <location> --asset_id <asset-id>
"""
import argparse
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceClient
from google.protobuf import empty_pb2 as empty

def delete_asset(project_id: str, location: str, asset_id: str) -> empty.Empty:
    if False:
        while True:
            i = 10
    'Deletes an asset.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the asset.\n        asset_id: The user-defined asset ID.'
    client = LivestreamServiceClient()
    name = f'projects/{project_id}/locations/{location}/assets/{asset_id}'
    operation = client.delete_asset(name=name)
    response = operation.result(600)
    print('Deleted asset')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the asset.', required=True)
    parser.add_argument('--asset_id', help='The user-defined asset ID.', required=True)
    args = parser.parse_args()
    delete_asset(args.project_id, args.location, args.asset_id)