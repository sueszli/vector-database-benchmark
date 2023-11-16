import re
from typing import Optional
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import RetryError
from google.cloud import documentai
from google.cloud import storage

def batch_process_documents(project_id: str, location: str, processor_id: str, gcs_output_uri: str, processor_version_id: Optional[str]=None, gcs_input_uri: Optional[str]=None, input_mime_type: Optional[str]=None, gcs_input_prefix: Optional[str]=None, field_mask: Optional[str]=None, timeout: int=400) -> None:
    if False:
        for i in range(10):
            print('nop')
    opts = ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com')
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    if gcs_input_uri:
        gcs_document = documentai.GcsDocument(gcs_uri=gcs_input_uri, mime_type=input_mime_type)
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    else:
        gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=gcs_input_prefix)
        input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)
    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=gcs_output_uri, field_mask=field_mask)
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)
    if processor_version_id:
        name = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    else:
        name = client.processor_path(project_id, location, processor_id)
    request = documentai.BatchProcessRequest(name=name, input_documents=input_config, document_output_config=output_config)
    operation = client.batch_process_documents(request)
    try:
        print(f'Waiting for operation {operation.operation.name} to complete...')
        operation.result(timeout=timeout)
    except (RetryError, InternalServerError) as e:
        print(e.message)
    metadata = documentai.BatchProcessMetadata(operation.metadata)
    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f'Batch Process Failed: {metadata.state_message}')
    storage_client = storage.Client()
    print('Output files:')
    for process in list(metadata.individual_process_statuses):
        matches = re.match('gs://(.*?)/(.*)', process.output_gcs_destination)
        if not matches:
            print('Could not parse output GCS destination:', process.output_gcs_destination)
            continue
        (output_bucket, output_prefix) = matches.groups()
        output_blobs = storage_client.list_blobs(output_bucket, prefix=output_prefix)
        for blob in output_blobs:
            if blob.content_type != 'application/json':
                print(f'Skipping non-supported file: {blob.name} - Mimetype: {blob.content_type}')
                continue
            print(f'Fetching {blob.name}')
            document = documentai.Document.from_json(blob.download_as_bytes(), ignore_unknown_fields=True)
            print('The document contains the following text:')
            print(document.text)