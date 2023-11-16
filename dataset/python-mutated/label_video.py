import argparse
import os
from google.api_core.client_options import ClientOptions

def label_video(dataset_resource_name, instruction_resource_name, annotation_spec_set_resource_name):
    if False:
        print('Hello World!')
    'Labels a video dataset.'
    from google.cloud import datalabeling_v1beta1 as datalabeling
    client = datalabeling.DataLabelingServiceClient()
    if 'DATALABELING_ENDPOINT' in os.environ:
        opts = ClientOptions(api_endpoint=os.getenv('DATALABELING_ENDPOINT'))
        client = datalabeling.DataLabelingServiceClient(client_options=opts)
    basic_config = datalabeling.HumanAnnotationConfig(instruction=instruction_resource_name, annotated_dataset_display_name='YOUR_ANNOTATED_DATASET_DISPLAY_NAME', label_group='YOUR_LABEL_GROUP', replica_count=1)
    feature = datalabeling.LabelVideoRequest.Feature.OBJECT_TRACKING
    config = datalabeling.ObjectTrackingConfig(annotation_spec_set=annotation_spec_set_resource_name)
    response = client.label_video(request={'parent': dataset_resource_name, 'basic_config': basic_config, 'feature': feature, 'object_tracking_config': config})
    print(f'Label_video operation name: {response.operation.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-resource-name', help='Dataset resource name. Required.', required=True)
    parser.add_argument('--instruction-resource-name', help='Instruction resource name. Required.', required=True)
    parser.add_argument('--annotation-spec-set-resource-name', help='Annotation spec set resource name. Required.', required=True)
    args = parser.parse_args()
    label_video(args.dataset_resource_name, args.instruction_resource_name, args.annotation_spec_set_resource_name)