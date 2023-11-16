import pytest
from apistar.core import validate
import typesystem

def test_validate_openapi():
    if False:
        i = 10
        return i + 15
    schema = '\n    {\n        "openapi": "3.0.0",\n        "info": {"title": "", "version": ""},\n        "paths": {}\n    }\n    '
    validate(schema, format='openapi', encoding='json')

def test_validate_openapi_datastructure():
    if False:
        i = 10
        return i + 15
    schema = {'openapi': '3.0.0', 'info': {'title': '', 'version': ''}, 'paths': {}}
    validate(schema, format='openapi')

def test_validate_autodetermine_openapi():
    if False:
        for i in range(10):
            print('nop')
    schema = '\n    {\n        "openapi": "3.0.0",\n        "info": {"title": "", "version": ""},\n        "paths": {}\n    }\n    '
    validate(schema, encoding='json')

def test_validate_autodetermine_swagger():
    if False:
        for i in range(10):
            print('nop')
    schema = '\n    {\n        "swagger": "2.0",\n        "info": {"title": "", "version": ""},\n        "paths": {}\n    }\n    '
    validate(schema, encoding='json')

def test_validate_autodetermine_failed():
    if False:
        while True:
            i = 10
    schema = '\n    {\n        "info": {"title": "", "version": ""},\n        "paths": {}\n    }\n    '
    with pytest.raises(typesystem.ValidationError):
        validate(schema, encoding='json')

def test_validate_with_bad_format():
    if False:
        return 10
    schema = '\n    {\n        "openapi": "3.0.0",\n        "info": {"title": "", "version": ""},\n        "paths": {}\n    }\n    '
    with pytest.raises(ValueError):
        validate(schema, format='xxx')

def test_validate_with_bad_encoding():
    if False:
        return 10
    schema = '\n    {\n        "openapi": "3.0.0",\n        "info": {"title": "", "version": ""},\n        "paths": {}\n    }\n    '
    with pytest.raises(ValueError):
        validate(schema, format='openapi', encoding='xxx')

def test_infer_json():
    if False:
        print('Hello World!')
    "\n    If 'encoding=' is omitted, then it should inferred from the content.\n    "
    schema = '\n    {\n        "openapi": "3.0.0",\n        "info": {"title": "", "version": ""},\n        "paths": {}\n    }\n    '
    validate(schema, format='openapi')

def test_infer_yaml():
    if False:
        for i in range(10):
            print('nop')
    "\n    If 'encoding=' is omitted, then it should inferred from the content.\n    "
    schema = '\n        openapi: "3.0.0"\n        info:\n            title: ""\n            version: ""\n        paths: {}\n    '
    validate(schema, format='openapi')