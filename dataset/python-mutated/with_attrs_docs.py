from typing import Type
from class_doc import extract_docs_from_cls_obj
from pydantic import BaseModel

def apply_attributes_docs(model: Type[BaseModel], override_existing: bool=True) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Apply model attributes documentation in-place. Resulted docs are placed\n    inside :code:`field.schema.description` for *pydantic* model field.\n    :param model: any pydantic model\n    :param override_existing: override existing descriptions\n    '
    docs = extract_docs_from_cls_obj(model)
    for field in model.__fields__.values():
        if field.field_info.description and (not override_existing):
            continue
        try:
            field.field_info.description = '\n'.join(docs[field.name])
        except KeyError:
            pass

def with_attrs_docs(model_cls: Type[BaseModel]) -> Type[BaseModel]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Applies :py:func:`.apply_attributes_docs`.\n    '

    def decorator(maybe_model_cls: Type[BaseModel]) -> Type[BaseModel]:
        if False:
            for i in range(10):
                print('nop')
        apply_attributes_docs(maybe_model_cls)
        return maybe_model_cls
    return decorator(model_cls)