from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def evaluate_processor_version_sample(project_id: str, location: str, processor_id: str, processor_version_id: str, gcs_input_uri: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    evaluation_documents = documentai.BatchDocumentsInputConfig(gcs_prefix=documentai.GcsPrefix(gcs_uri_prefix=gcs_input_uri))
    request = documentai.EvaluateProcessorVersionRequest(processor_version=name, evaluation_documents=evaluation_documents)
    operation = client.evaluate_processor_version(request=request)
    print(f'Waiting for operation {operation.operation.name} to complete...')
    response = documentai.EvaluateProcessorVersionResponse(operation.result())
    print(f'Evaluation Complete: {response.evaluation}')