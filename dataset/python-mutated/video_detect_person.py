import io
from google.cloud import videointelligence_v1 as videointelligence

def detect_person(local_file_path='path/to/your/video-file.mp4'):
    if False:
        i = 10
        return i + 15
    'Detects people in a video from a local file.'
    client = videointelligence.VideoIntelligenceServiceClient()
    with io.open(local_file_path, 'rb') as f:
        input_content = f.read()
    config = videointelligence.types.PersonDetectionConfig(include_bounding_boxes=True, include_attributes=True, include_pose_landmarks=True)
    context = videointelligence.types.VideoContext(person_detection_config=config)
    operation = client.annotate_video(request={'features': [videointelligence.Feature.PERSON_DETECTION], 'input_content': input_content, 'video_context': context})
    print('\nProcessing video for person detection annotations.')
    result = operation.result(timeout=300)
    print('\nFinished processing.\n')
    annotation_result = result.annotation_results[0]
    for annotation in annotation_result.person_detection_annotations:
        print('Person detected:')
        for track in annotation.tracks:
            print('Segment: {}s to {}s'.format(track.segment.start_time_offset.seconds + track.segment.start_time_offset.microseconds / 1000000.0, track.segment.end_time_offset.seconds + track.segment.end_time_offset.microseconds / 1000000.0))
            timestamped_object = track.timestamped_objects[0]
            box = timestamped_object.normalized_bounding_box
            print('Bounding box:')
            print('\tleft  : {}'.format(box.left))
            print('\ttop   : {}'.format(box.top))
            print('\tright : {}'.format(box.right))
            print('\tbottom: {}'.format(box.bottom))
            print('Attributes:')
            for attribute in timestamped_object.attributes:
                print('\t{}:{} {}'.format(attribute.name, attribute.value, attribute.confidence))
            print('Landmarks:')
            for landmark in timestamped_object.landmarks:
                print('\t{}: {} (x={}, y={})'.format(landmark.name, landmark.confidence, landmark.point.x, landmark.point.y))