from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def list_evaluations_sample(project_id: str, location: str, processor_id: str, processor_version_id: str) -> None:
    if False:
        i = 10
        return i + 15
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    parent = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    evaluations = client.list_evaluations(parent=parent)
    print(f'Evaluations for Processor Version {parent}')
    for evaluation in evaluations:
        print(f'Name: {evaluation.name}')
        print(f'\tCreate Time: {evaluation.create_time}\n')