"""Google Cloud Video Stitcher sample for creating a live stream session in
which to insert ads.
Example usage:
    python create_live_session.py --project_id <project-id>         --location <location> --live_config_id <live-config-id>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def create_live_session(project_id: str, location: str, live_config_id: str) -> stitcher_v1.types.LiveSession:
    if False:
        for i in range(10):
            print('nop')
    'Creates a live session. Live sessions are ephemeral resources that expire\n    after a few minutes.\n    Args:\n        project_id: The GCP project ID.\n        location: The location in which to create the session.\n        live_config_id: The user-defined live config ID.\n\n    Returns:\n        The live session resource.\n    '
    client = VideoStitcherServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    live_config = f'projects/{project_id}/locations/{location}/liveConfigs/{live_config_id}'
    live_session = stitcher_v1.types.LiveSession(live_config=live_config)
    response = client.create_live_session(parent=parent, live_session=live_session)
    print(f'Live session: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location in which to create the live session.', default='us-central1')
    parser.add_argument('--live_config_id', help='The user-defined live config ID.', required=True)
    args = parser.parse_args()
    create_live_session(args.project_id, args.location, args.live_config_id)