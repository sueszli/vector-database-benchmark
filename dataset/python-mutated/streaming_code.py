import vertexai
from vertexai import language_models

def streaming_prediction(project_id: str, location: str) -> str:
    if False:
        return 10
    'Streaming Code Example with a Large Language Model.'
    vertexai.init(project=project_id, location=location)
    code_generation_model = language_models.CodeGenerationModel.from_pretrained('code-bison')
    parameters = {'temperature': 0.8, 'max_output_tokens': 256}
    responses = code_generation_model.predict_streaming(prefix='Write a function that checks if a year is a leap year.', **parameters)
    results = []
    for response in responses:
        print(response)
        results.append(str(response))
    results = '\n'.join(results)
    return results
if __name__ == '__main__':
    streaming_prediction()