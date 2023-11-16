"""Google Cloud Video Stitcher sample for listing all live configs in a location.
Example usage:
    python list_live_configs.py --project_id <project-id> --location <location>
"""
import argparse
from google.cloud.video.stitcher_v1.services.video_stitcher_service import pagers, VideoStitcherServiceClient

def list_live_configs(project_id: str, location: str) -> pagers.ListLiveConfigsPager:
    if False:
        return 10
    'Lists all live configs in a location.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the live configs.\n\n    Returns:\n        An iterable object containing live config resources.\n    '
    client = VideoStitcherServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    response = client.list_live_configs(parent=parent)
    print('Live configs:')
    for live_config in response.live_configs:
        print({live_config.name})
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the live configs.', required=True)
    args = parser.parse_args()
    list_live_configs(args.project_id, args.location)