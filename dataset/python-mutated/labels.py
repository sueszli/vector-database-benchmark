"""This application demonstrates how to detect labels from a video
based on the image content with the Google Cloud Video Intelligence
API.

For more information, check out the documentation at
https://cloud.google.com/videointelligence/docs.

Usage Example:

    python labels.py gs://cloud-ml-sandbox/video/chicago.mp4

"""
import argparse
from google.cloud import videointelligence

def analyze_labels(path):
    if False:
        for i in range(10):
            print('nop')
    'Detects labels given a GCS path.'
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.LABEL_DETECTION]
    operation = video_client.annotate_video(request={'features': features, 'input_uri': path})
    print('\nProcessing video for label annotations:')
    result = operation.result(timeout=90)
    print('\nFinished processing.')
    segment_labels = result.annotation_results[0].segment_label_annotations
    for (i, segment_label) in enumerate(segment_labels):
        print('Video label description: {}'.format(segment_label.entity.description))
        for category_entity in segment_label.category_entities:
            print('\tLabel category description: {}'.format(category_entity.description))
        for (i, segment) in enumerate(segment_label.segments):
            start_time = segment.segment.start_time_offset.seconds + segment.segment.start_time_offset.microseconds / 1000000.0
            end_time = segment.segment.end_time_offset.seconds + segment.segment.end_time_offset.microseconds / 1000000.0
            positions = '{}s to {}s'.format(start_time, end_time)
            confidence = segment.confidence
            print('\tSegment {}: {}'.format(i, positions))
            print('\tConfidence: {}'.format(confidence))
        print('\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='GCS file path for label detection.')
    args = parser.parse_args()
    analyze_labels(args.path)