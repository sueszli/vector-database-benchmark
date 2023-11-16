from google.cloud import automl_v1

def sample_get_annotation_spec():
    if False:
        i = 10
        return i + 15
    client = automl_v1.AutoMlClient()
    request = automl_v1.GetAnnotationSpecRequest(name='name_value')
    response = client.get_annotation_spec(request=request)
    print(response)