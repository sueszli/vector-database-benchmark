from google.cloud import videointelligence_v1 as videointelligence

def detect_faces(gcs_uri='gs://YOUR_BUCKET_ID/path/to/your/video.mp4'):
    if False:
        return 10
    'Detects faces in a video.'
    client = videointelligence.VideoIntelligenceServiceClient()
    config = videointelligence.FaceDetectionConfig(include_bounding_boxes=True, include_attributes=True)
    context = videointelligence.VideoContext(face_detection_config=config)
    operation = client.annotate_video(request={'features': [videointelligence.Feature.FACE_DETECTION], 'input_uri': gcs_uri, 'video_context': context})
    print('\nProcessing video for face detection annotations.')
    result = operation.result(timeout=300)
    print('\nFinished processing.\n')
    annotation_result = result.annotation_results[0]
    for annotation in annotation_result.face_detection_annotations:
        print('Face detected:')
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