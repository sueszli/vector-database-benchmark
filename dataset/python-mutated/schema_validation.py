import pytest
from jsonschema import ValidationError

def invalid_schema(func):
    if False:
        print('Hello World!')

    def inner(self, *args, **kwargs):
        if False:
            return 10
        with pytest.raises(ValidationError):
            func(self)
    return inner

def invalid_schema_with_error_message(message):
    if False:
        i = 10
        return i + 15

    def decorator(func):
        if False:
            for i in range(10):
                print('nop')

        def inner(self, *args, **kwargs):
            if False:
                return 10
            with pytest.raises(ValidationError) as excinfo:
                func(self)
            assert excinfo.value.message == message
        return inner
    return decorator