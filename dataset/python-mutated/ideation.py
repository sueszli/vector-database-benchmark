import vertexai
from vertexai.language_models import TextGenerationModel

def interview(temperature: float, project_id: str, location: str) -> str:
    if False:
        i = 10
        return i + 15
    'Ideation example with a Large Language Model'
    vertexai.init(project=project_id, location=location)
    parameters = {'temperature': temperature, 'max_output_tokens': 256, 'top_p': 0.8, 'top_k': 40}
    model = TextGenerationModel.from_pretrained('text-bison@001')
    response = model.predict('Give me ten interview questions for the role of program manager.', **parameters)
    print(f'Response from Model: {response.text}')
    return response.text
if __name__ == '__main__':
    interview()