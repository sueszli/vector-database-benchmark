import argparse
import os
from google.api_core.client_options import ClientOptions

def create_instruction(project_id, data_type, instruction_gcs_uri):
    if False:
        i = 10
        return i + 15
    'Creates a data labeling PDF instruction for the given Google Cloud\n    project. The PDF file should be uploaded to the project in\n    Google Cloud Storage.\n    '
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    project_path = f'projects/{project_id}'
    pdf_instruction = datalabeling.PdfInstruction(gcs_file_uri=instruction_gcs_uri)
    instruction = datalabeling.Instruction(display_name='YOUR_INSTRUCTION_DISPLAY_NAME', description='YOUR_DESCRIPTION', data_type=data_type, pdf_instruction=pdf_instruction)
    operation = client.create_instruction(request={'parent': project_path, 'instruction': instruction})
    result = operation.result()
    print(f'The instruction resource name: {result.name}')
    print(f'Display name: {result.display_name}')
    print(f'Description: {result.description}')
    print('Create time:')
    print(f'\tseconds: {result.create_time.timestamp_pb().seconds}')
    print(f'\tnanos: {result.create_time.timestamp_pb().nanos}')
    print(f'Data type: {datalabeling.DataType(result.data_type).name}')
    print('Pdf instruction:')
    print(f'\tGcs file uri: {result.pdf_instruction.gcs_file_uri}\n')
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-id', help='Project ID. Required.', required=True)
    parser.add_argument('--data-type', help='Data type. Only support IMAGE, VIDEO, TEXT and AUDIO. Required.', required=True)
    parser.add_argument('--instruction-gcs-uri', help='The URI of Google Cloud Storage of the instruction. Required.', required=True)
    args = parser.parse_args()
    create_instruction(args.project_id, args.data_type, args.instruction_gcs_uri)