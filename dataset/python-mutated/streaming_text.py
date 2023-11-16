import vertexai
from vertexai import language_models

def streaming_prediction(project_id: str, location: str) -> str:
    if False:
        while True:
            i = 10
    'Streaming Text Example with a Large Language Model.'
    vertexai.init(project=project_id, location=location)
    text_generation_model = language_models.TextGenerationModel.from_pretrained('text-bison')
    parameters = {'temperature': 0.2, 'max_output_tokens': 256, 'top_p': 0.8, 'top_k': 40}
    responses = text_generation_model.predict_streaming(prompt='Give me ten interview questions for the role of program manager.', **parameters)
    results = []
    for response in responses:
        print(response)
        results.append(str(response))
    results = '\n'.join(results)
    return results
if __name__ == '__main__':
    streaming_prediction()