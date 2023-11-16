def analyze_image():
    if False:
        while True:
            i = 10
    import os
    from azure.ai.contentsafety import ContentSafetyClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
    key = os.environ['CONTENT_SAFETY_KEY']
    endpoint = os.environ['CONTENT_SAFETY_ENDPOINT']
    image_path = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', './sample_data/image.jpg'))
    client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
    with open(image_path, 'rb') as file:
        request = AnalyzeImageOptions(image=ImageData(content=file.read()))
    try:
        response = client.analyze_image(request)
    except HttpResponseError as e:
        print('Analyze image failed.')
        if e.error:
            print(f'Error code: {e.error.code}')
            print(f'Error message: {e.error.message}')
            raise
        print(e)
        raise
    if response.hate_result:
        print(f'Hate severity: {response.hate_result.severity}')
    if response.self_harm_result:
        print(f'SelfHarm severity: {response.self_harm_result.severity}')
    if response.sexual_result:
        print(f'Sexual severity: {response.sexual_result.severity}')
    if response.violence_result:
        print(f'Violence severity: {response.violence_result.severity}')
if __name__ == '__main__':
    analyze_image()