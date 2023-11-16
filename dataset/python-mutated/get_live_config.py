"""Google Cloud Video Stitcher sample for getting a live config.
Example usage:
    python get_live_config.py --project_id <project-id> --location <location>         --live_config_id <live-config-id>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def get_live_config(project_id: str, location: str, live_config_id: str) -> stitcher_v1.types.LiveConfig:
    if False:
        return 10
    'Gets a live config.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the live config.\n        live_config_id: The user-defined live config ID.\n\n    Returns:\n        The live config resource.\n    '
    client = VideoStitcherServiceClient()
    name = f'projects/{project_id}/locations/{location}/liveConfigs/{live_config_id}'
    response = client.get_live_config(name=name)
    print(f'Live config: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the live config.', required=True)
    parser.add_argument('--live_config_id', help='The user-defined live config ID.', required=True)
    args = parser.parse_args()
    get_live_config(args.project_id, args.location, args.live_config_id)