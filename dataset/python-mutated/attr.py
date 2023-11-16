from botocore.docs.params import ResponseParamsDocumenter
from boto3.docs.utils import get_identifier_description

class ResourceShapeDocumenter(ResponseParamsDocumenter):
    EVENT_NAME = 'resource-shape'

def document_attribute(section, service_name, resource_name, attr_name, event_emitter, attr_model, include_signature=True):
    if False:
        i = 10
        return i + 15
    if include_signature:
        full_attr_name = f"{section.context.get('qualifier', '')}{attr_name}"
        section.style.start_sphinx_py_attr(full_attr_name)
    ResourceShapeDocumenter(service_name=service_name, operation_name=resource_name, event_emitter=event_emitter).document_params(section=section, shape=attr_model)

def document_identifier(section, resource_name, identifier_model, include_signature=True):
    if False:
        i = 10
        return i + 15
    if include_signature:
        full_identifier_name = f"{section.context.get('qualifier', '')}{identifier_model.name}"
        section.style.start_sphinx_py_attr(full_identifier_name)
    description = get_identifier_description(resource_name, identifier_model.name)
    section.write(f'*(string)* {description}')

def document_reference(section, reference_model, include_signature=True):
    if False:
        while True:
            i = 10
    if include_signature:
        full_reference_name = f"{section.context.get('qualifier', '')}{reference_model.name}"
        section.style.start_sphinx_py_attr(full_reference_name)
    reference_type = f'(:py:class:`{reference_model.resource.type}`) '
    section.write(reference_type)
    section.include_doc_string(f'The related {reference_model.name} if set, otherwise ``None``.')