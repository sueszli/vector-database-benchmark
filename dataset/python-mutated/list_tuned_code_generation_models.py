import vertexai
from vertexai.preview.language_models import CodeGenerationModel

def list_tuned_code_generation_models(project_id: str, location: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'List tuned models.'
    vertexai.init(project=project_id, location=location)
    model = CodeGenerationModel.from_pretrained('code-bison@001')
    tuned_model_names = model.list_tuned_model_names()
    print(tuned_model_names)
    return tuned_model_names
if __name__ == '__main__':
    list_tuned_code_generation_models()