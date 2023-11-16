"""Google Cloud Video Stitcher sample for getting a video on demand (VOD)
session.
Example usage:
    python get_vod_session.py --project_id <project-id> --location <location>         --session_id <session-id>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def get_vod_session(project_id: str, location: str, session_id: str) -> stitcher_v1.types.VodSession:
    if False:
        for i in range(10):
            print('nop')
    'Gets a VOD session. VOD sessions are ephemeral resources that expire\n    after a few hours.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the session.\n        session_id: The ID of the VOD session.\n\n    Returns:\n        The VOD session resource.\n    '
    client = VideoStitcherServiceClient()
    name = client.vod_session_path(project_id, location, session_id)
    response = client.get_vod_session(name=name)
    print(f'VOD session: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the VOD session.', required=True)
    parser.add_argument('--session_id', help='The ID of the VOD session.', required=True)
    args = parser.parse_args()
    get_vod_session(args.project_id, args.location, args.session_id)