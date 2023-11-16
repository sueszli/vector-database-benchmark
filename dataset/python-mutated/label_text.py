import argparse
import os
from google.api_core.client_options import ClientOptions

def label_text(dataset_resource_name, instruction_resource_name, annotation_spec_set_resource_name):
    if False:
        while True:
            i = 10
    'Labels a text dataset.'
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    basic_config = datalabeling.HumanAnnotationConfig(instruction=instruction_resource_name, annotated_dataset_display_name='YOUR_ANNOTATED_DATASET_DISPLAY_NAME', label_group='YOUR_LABEL_GROUP', replica_count=1)
    feature = datalabeling.LabelTextRequest.Feature.TEXT_ENTITY_EXTRACTION
    config = datalabeling.TextEntityExtractionConfig(annotation_spec_set=annotation_spec_set_resource_name)
    response = client.label_text(request={'parent': dataset_resource_name, 'basic_config': basic_config, 'feature': feature, 'text_classification_config': config})
    print(f'Label_text operation name: {response.operation.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-resource-name', help='Dataset resource name. Required.', required=True)
    parser.add_argument('--instruction-resource-name', help='Instruction resource name. Required.', required=True)
    parser.add_argument('--annotation-spec-set-resource-name', help='Annotation spec set resource name. Required.', required=True)
    args = parser.parse_args()
    label_text(args.dataset_resource_name, args.instruction_resource_name, args.annotation_spec_set_resource_name)