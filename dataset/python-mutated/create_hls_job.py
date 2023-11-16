import boto3
from botocore.exceptions import ClientError

def create_elastic_transcoder_hls_job(pipeline_id, input_file, outputs, output_file_prefix, playlists):
    if False:
        while True:
            i = 10
    "Create an Elastic Transcoder HSL job\n\n    :param pipeline_id: string; ID of an existing Elastic Transcoder pipeline\n    :param input_file: string; Name of existing object in pipeline's S3 input bucket\n    :param outputs: list of dictionaries; Parameters defining each output file\n    :param output_file_prefix: string; Prefix for each output file name\n    :param playlists: list of dictionaries; Parameters defining each playlist\n    :return Dictionary containing information about the job\n            If job could not be created, returns None\n    "
    etc_client = boto3.client('elastictranscoder')
    try:
        response = etc_client.create_job(PipelineId=pipeline_id, Input={'Key': input_file}, Outputs=outputs, OutputKeyPrefix=output_file_prefix, Playlists=playlists)
    except ClientError as e:
        print(f'ERROR: {e}')
        return None
    return response['Job']

def main():
    if False:
        print('Hello World!')
    'Exercise Elastic Transcoder create_job operation\n\n    Before running this script, all Elastic Transcoder setup must be\n    completed, such as defining the pipeline and specifying the S3 input\n    and output buckets. Also, the file to transcode must exist in the S3\n    input bucket.\n    '
    pipeline_id = 'PIPELINE_ID'
    input_file = 'FILE_TO_TRANSCODE'
    output_file = 'TRANSCODED_FILE'
    output_file_prefix = 'elastic-transcoder-samples/output/hls/'
    segment_duration = '2'
    hls_64k_audio_preset_id = '1351620000001-200071'
    hls_0400k_preset_id = '1351620000001-200050'
    hls_0600k_preset_id = '1351620000001-200040'
    hls_1000k_preset_id = '1351620000001-200030'
    hls_1500k_preset_id = '1351620000001-200020'
    hls_2000k_preset_id = '1351620000001-200010'
    outputs = [{'Key': 'hlsAudio/' + output_file, 'PresetId': hls_64k_audio_preset_id, 'SegmentDuration': segment_duration}, {'Key': 'hls0400k/' + output_file, 'PresetId': hls_0400k_preset_id, 'SegmentDuration': segment_duration}, {'Key': 'hls0600k/' + output_file, 'PresetId': hls_0600k_preset_id, 'SegmentDuration': segment_duration}, {'Key': 'hls1000k/' + output_file, 'PresetId': hls_1000k_preset_id, 'SegmentDuration': segment_duration}, {'Key': 'hls1500k/' + output_file, 'PresetId': hls_1500k_preset_id, 'SegmentDuration': segment_duration}, {'Key': 'hls2000k/' + output_file, 'PresetId': hls_2000k_preset_id, 'SegmentDuration': segment_duration}]
    playlists = [{'Name': 'hls_' + output_file, 'Format': 'HLSv3', 'OutputKeys': [x['Key'] for x in outputs]}]
    job_info = create_elastic_transcoder_hls_job(pipeline_id, input_file, outputs, output_file_prefix, playlists)
    if job_info is None:
        exit(1)
    print(f"Created Amazon Elastic Transcoder HLS job {job_info['Id']}")
if __name__ == '__main__':
    main()