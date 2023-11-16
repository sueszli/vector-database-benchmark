from google.cloud import speech_v1p1beta1

def sample_list_custom_classes():
    if False:
        for i in range(10):
            print('nop')
    client = speech_v1p1beta1.AdaptationClient()
    request = speech_v1p1beta1.ListCustomClassesRequest(parent='parent_value')
    page_result = client.list_custom_classes(request=request)
    for response in page_result:
        print(response)