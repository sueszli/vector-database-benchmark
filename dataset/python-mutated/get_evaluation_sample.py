from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def get_evaluation_sample(project_id: str, location: str, processor_id: str, processor_version_id: str, evaluation_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    evaluation_name = client.evaluation_path(project_id, location, processor_id, processor_version_id, evaluation_id)
    evaluation = client.get_evaluation(name=evaluation_name)
    create_time = evaluation.create_time
    document_counters = evaluation.document_counters
    print(f'Create Time: {create_time}')
    print(f'Input Documents: {document_counters.input_documents_count}')
    print(f'\tInvalid Documents: {document_counters.invalid_documents_count}')
    print(f'\tFailed Documents: {document_counters.failed_documents_count}')
    print(f'\tEvaluated Documents: {document_counters.evaluated_documents_count}')