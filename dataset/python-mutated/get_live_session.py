"""Google Cloud Video Stitcher sample for getting a live stream session.
Example usage:
    python get_live_session.py --project_id <project-id> --location <location>         --session_id <session-id>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def get_live_session(project_id: str, location: str, session_id: str) -> stitcher_v1.types.LiveSession:
    if False:
        print('Hello World!')
    'Gets a live session. Live sessions are ephemeral resources that expire\n    after a few minutes.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the session.\n        session_id: The ID of the live session.\n\n    Returns:\n        The live session resource.\n    '
    client = VideoStitcherServiceClient()
    name = client.live_session_path(project_id, location, session_id)
    response = client.get_live_session(name=name)
    print(f'Live session: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the live session.', required=True)
    parser.add_argument('--session_id', help='The ID of the live session.', required=True)
    args = parser.parse_args()
    get_live_session(args.project_id, args.location, args.session_id)