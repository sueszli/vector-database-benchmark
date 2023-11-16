from typing import Optional
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

def train_processor_version_sample(project_id: str, location: str, processor_id: str, processor_version_display_name: str, train_data_uri: Optional[str]=None, test_data_uri: Optional[str]=None) -> None:
    if False:
        print('Hello World!')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    parent = client.processor_path(project_id, location, processor_id)
    processor_version = documentai.ProcessorVersion(display_name=processor_version_display_name)
    input_data = documentai.TrainProcessorVersionRequest.InputData(training_documents=documentai.BatchDocumentsInputConfig(gcs_prefix=documentai.GcsPrefix(gcs_uri_prefix=train_data_uri)), test_documents=documentai.BatchDocumentsInputConfig(gcs_prefix=documentai.GcsPrefix(gcs_uri_prefix=test_data_uri)))
    request = documentai.TrainProcessorVersionRequest(parent=parent, processor_version=processor_version, input_data=input_data)
    operation = client.train_processor_version(request=request)
    print(operation.operation.name)
    response = documentai.TrainProcessorVersionResponse(operation.result())
    metadata = documentai.TrainProcessorVersionMetadata(operation.metadata)
    print(f'New Processor Version:{response.processor_version}')
    print(f'Training Set Validation: {metadata.training_dataset_validation}')
    print(f'Test Set Validation: {metadata.test_dataset_validation}')