"""This application demonstrates how to identify all different shots
in a video using the Google Cloud Video Intelligence API.

For more information, check out the documentation at
https://cloud.google.com/videointelligence/docs.

Example Usage:

    python shotchange.py gs://cloud-samples-data/video/gbikes_dinosaur.mp4

"""
import argparse
from google.cloud import videointelligence

def analyze_shots(path):
    if False:
        i = 10
        return i + 15
    'Detects camera shot changes.'
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
    operation = video_client.annotate_video(request={'features': features, 'input_uri': path})
    print('\nProcessing video for shot change annotations:')
    result = operation.result(timeout=120)
    print('\nFinished processing.')
    for (i, shot) in enumerate(result.annotation_results[0].shot_annotations):
        start_time = shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1000000.0
        end_time = shot.end_time_offset.seconds + shot.end_time_offset.microseconds / 1000000.0
        print('\tShot {}: {} to {}'.format(i, start_time, end_time))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='GCS path for shot change detection.')
    args = parser.parse_args()
    analyze_shots(args.path)