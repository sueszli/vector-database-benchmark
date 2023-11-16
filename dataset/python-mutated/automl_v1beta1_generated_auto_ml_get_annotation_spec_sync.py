from google.cloud import automl_v1beta1

def sample_get_annotation_spec():
    if False:
        print('Hello World!')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.GetAnnotationSpecRequest(name='name_value')
    response = client.get_annotation_spec(request=request)
    print(response)