import vertexai
from vertexai.language_models import TextGenerationModel

def list_tuned_models(project_id: str, location: str) -> None:
    if False:
        while True:
            i = 10
    'List tuned models.'
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained('text-bison@001')
    tuned_model_names = model.list_tuned_model_names()
    print(tuned_model_names)
    return tuned_model_names
if __name__ == '__main__':
    list_tuned_models()