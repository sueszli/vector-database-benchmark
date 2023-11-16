import json
from pydantic import BaseModel

def _apply_default_pydantic_kwargs(kwargs: dict) -> dict:
    if False:
        while True:
            i = 10
    'A helper function to apply default kwargs to pydantic models.\n\n    Args:\n        kwargs (dict): the kwargs to apply\n\n    Returns:\n        dict: the kwargs with defaults applied\n    '
    default_kwargs = {'by_alias': True, 'exclude_none': True}
    return {**default_kwargs, **kwargs}

def to_json_sanitized_dict(pydantic_model_obj: BaseModel, **kwargs) -> dict:
    if False:
        return 10
    'A helper function to convert a pydantic model to a sanitized dict.\n\n    Without this pydantic dictionary may contain values that are not JSON serializable.\n\n    Args:\n        pydantic_model_obj (BaseModel): a pydantic model\n\n    Returns:\n        dict: a sanitized dictionary\n    '
    return json.loads(to_json(pydantic_model_obj, **kwargs))

def to_json(pydantic_model_obj: BaseModel, **kwargs) -> str:
    if False:
        print('Hello World!')
    'A helper function to convert a pydantic model to a json string.\n\n    Without this pydantic dictionary may contain values that are not JSON serializable.\n\n    Args:\n        pydantic_model_obj (BaseModel): a pydantic model\n\n    Returns:\n        str: a json string\n    '
    kwargs = _apply_default_pydantic_kwargs(kwargs)
    return pydantic_model_obj.json(**kwargs)

def to_dict(pydantic_model_obj: BaseModel, **kwargs) -> dict:
    if False:
        i = 10
        return i + 15
    'A helper function to convert a pydantic model to a dict.\n\n    Without this pydantic dictionary may contain values that are not JSON serializable.\n\n    Args:\n        pydantic_model_obj (BaseModel): a pydantic model\n\n    Returns:\n        dict: a dict\n    '
    kwargs = _apply_default_pydantic_kwargs(kwargs)
    return pydantic_model_obj.dict(**kwargs)