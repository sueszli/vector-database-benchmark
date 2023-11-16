from google.cloud import videointelligence

def detect_logo_gcs(input_uri='gs://YOUR_BUCKET_ID/path/to/your/file.mp4'):
    if False:
        while True:
            i = 10
    client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.LOGO_RECOGNITION]
    operation = client.annotate_video(request={'features': features, 'input_uri': input_uri})
    print('Waiting for operation to complete...')
    response = operation.result()
    annotation_result = response.annotation_results[0]
    for logo_recognition_annotation in annotation_result.logo_recognition_annotations:
        entity = logo_recognition_annotation.entity
        print('Entity Id : {}'.format(entity.entity_id))
        print('Description : {}'.format(entity.description))
        for track in logo_recognition_annotation.tracks:
            print('\n\tStart Time Offset : {}.{}'.format(track.segment.start_time_offset.seconds, track.segment.start_time_offset.microseconds * 1000))
            print('\tEnd Time Offset : {}.{}'.format(track.segment.end_time_offset.seconds, track.segment.end_time_offset.microseconds * 1000))
            print('\tConfidence : {}'.format(track.confidence))
            for timestamped_object in track.timestamped_objects:
                normalized_bounding_box = timestamped_object.normalized_bounding_box
                print('\n\t\tLeft : {}'.format(normalized_bounding_box.left))
                print('\t\tTop : {}'.format(normalized_bounding_box.top))
                print('\t\tRight : {}'.format(normalized_bounding_box.right))
                print('\t\tBottom : {}'.format(normalized_bounding_box.bottom))
                for attribute in timestamped_object.attributes:
                    print('\n\t\t\tName : {}'.format(attribute.name))
                    print('\t\t\tConfidence : {}'.format(attribute.confidence))
                    print('\t\t\tValue : {}'.format(attribute.value))
            for track_attribute in track.attributes:
                print('\n\t\tName : {}'.format(track_attribute.name))
                print('\t\tConfidence : {}'.format(track_attribute.confidence))
                print('\t\tValue : {}'.format(track_attribute.value))
        for segment in logo_recognition_annotation.segments:
            print('\n\tStart Time Offset : {}.{}'.format(segment.start_time_offset.seconds, segment.start_time_offset.microseconds * 1000))
            print('\tEnd Time Offset : {}.{}'.format(segment.end_time_offset.seconds, segment.end_time_offset.microseconds * 1000))