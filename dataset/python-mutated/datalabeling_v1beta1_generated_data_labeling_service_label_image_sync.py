from google.cloud import datalabeling_v1beta1

def sample_label_image():
    if False:
        print('Hello World!')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    image_classification_config = datalabeling_v1beta1.ImageClassificationConfig()
    image_classification_config.annotation_spec_set = 'annotation_spec_set_value'
    basic_config = datalabeling_v1beta1.HumanAnnotationConfig()
    basic_config.instruction = 'instruction_value'
    basic_config.annotated_dataset_display_name = 'annotated_dataset_display_name_value'
    request = datalabeling_v1beta1.LabelImageRequest(image_classification_config=image_classification_config, parent='parent_value', basic_config=basic_config, feature='SEGMENTATION')
    operation = client.label_image(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)