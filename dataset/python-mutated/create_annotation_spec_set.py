import argparse
import os
from google.api_core.client_options import ClientOptions

def create_annotation_spec_set(project_id):
    if False:
        return 10
    'Creates a data labeling annotation spec set for the given\n    Google Cloud project.\n    '
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    project_path = f'projects/{project_id}'
    annotation_spec_1 = datalabeling.AnnotationSpec(display_name='label_1', description='label_description_1')
    annotation_spec_2 = datalabeling.AnnotationSpec(display_name='label_2', description='label_description_2')
    annotation_spec_set = datalabeling.AnnotationSpecSet(display_name='YOUR_ANNOTATION_SPEC_SET_DISPLAY_NAME', description='YOUR_DESCRIPTION', annotation_specs=[annotation_spec_1, annotation_spec_2])
    response = client.create_annotation_spec_set(request={'parent': project_path, 'annotation_spec_set': annotation_spec_set})
    print(f'The annotation_spec_set resource name: {response.name}')
    print(f'Display name: {response.display_name}')
    print(f'Description: {response.description}')
    print('Annotation specs:')
    for annotation_spec in response.annotation_specs:
        print(f'\tDisplay name: {annotation_spec.display_name}')
        print(f'\tDescription: {annotation_spec.description}\n')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-id', help='Project ID. Required.', required=True)
    args = parser.parse_args()
    create_annotation_spec_set(args.project_id)