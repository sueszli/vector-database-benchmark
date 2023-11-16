from typing import Optional
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1beta3 as documentai

def process_document_summarizer_sample(project_id: str, location: str, processor_id: str, processor_version: str, file_path: str, mime_type: str) -> None:
    if False:
        print('Hello World!')
    summary_options = documentai.SummaryOptions(length=documentai.SummaryOptions.Length.BRIEF, format=documentai.SummaryOptions.Format.BULLETS)
    properties = [documentai.DocumentSchema.EntityType.Property(name='summary', value_type='string', occurence_type=documentai.DocumentSchema.EntityType.Property.OccurenceType.REQUIRED_ONCE, property_metadata=documentai.PropertyMetadata(field_extraction_metadata=documentai.FieldExtractionMetadata(summary_options=summary_options)))]
    process_options = documentai.ProcessOptions(schema_override=documentai.DocumentSchema(entity_types=[documentai.DocumentSchema.EntityType(name='summary_document_type', base_types=['document'], properties=properties)]))
    document = process_document(project_id, location, processor_id, processor_version, file_path, mime_type, process_options=process_options)
    for entity in document.entities:
        print_entity(entity)
        for prop in entity.properties:
            print_entity(prop)

def process_document_custom_extractor_sample(project_id: str, location: str, processor_id: str, processor_version: str, file_path: str, mime_type: str) -> None:
    if False:
        while True:
            i = 10
    properties = [documentai.DocumentSchema.EntityType.Property(name='invoice_id', value_type='string', occurence_type=documentai.DocumentSchema.EntityType.Property.OccurenceType.REQUIRED_ONCE), documentai.DocumentSchema.EntityType.Property(name='notes', value_type='string', occurence_type=documentai.DocumentSchema.EntityType.Property.OccurenceType.REQUIRED_ONCE), documentai.DocumentSchema.EntityType.Property(name='terms', value_type='string', occurence_type=documentai.DocumentSchema.EntityType.Property.OccurenceType.REQUIRED_ONCE)]
    process_options = documentai.ProcessOptions(schema_override=documentai.DocumentSchema(display_name='CDE Schema', description='Document Schema for the CDE Processor', entity_types=[documentai.DocumentSchema.EntityType(name='custom_extraction_document_type', base_types=['document'], properties=properties)]))
    document = process_document(project_id, location, processor_id, processor_version, file_path, mime_type, process_options=process_options)
    for entity in document.entities:
        print_entity(entity)
        for prop in entity.properties:
            print_entity(prop)

def print_entity(entity: documentai.Document.Entity) -> None:
    if False:
        i = 10
        return i + 15
    key = entity.type_
    text_value = entity.text_anchor.content
    confidence = entity.confidence
    normalized_value = entity.normalized_value.text
    print(f'    * {repr(key)}: {repr(text_value)}({confidence:.1%} confident)')
    if normalized_value:
        print(f'    * Normalized Value: {repr(normalized_value)}')

def process_document(project_id: str, location: str, processor_id: str, processor_version: str, file_path: str, mime_type: str, process_options: Optional[documentai.ProcessOptions]=None) -> documentai.Document:
    if False:
        while True:
            i = 10
    client = documentai.DocumentProcessorServiceClient(client_options=ClientOptions(api_endpoint=f'{location}-documentai.googleapis.com'))
    name = client.processor_version_path(project_id, location, processor_id, processor_version)
    with open(file_path, 'rb') as image:
        image_content = image.read()
    request = documentai.ProcessRequest(name=name, raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type), process_options=process_options)
    result = client.process_document(request=request)
    return result.document