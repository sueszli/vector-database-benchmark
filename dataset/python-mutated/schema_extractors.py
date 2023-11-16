"""Helpers to retrieve schema from voluptuous validators.

These are a helper decorators to help get schema from some
components which uses voluptuous in a way where validation
is hidden in local functions
These decorators should not modify at all what the functions
originally do.
However there is a property to further disable decorator
impact."""
EnableSchemaExtraction = False
extended_schemas = {}
list_schemas = {}
registry_schemas = {}
hidden_schemas = {}
typed_schemas = {}
SCHEMA_EXTRACT = object()

def schema_extractor(validator_name):
    if False:
        print('Hello World!')
    if EnableSchemaExtraction:

        def decorator(func):
            if False:
                return 10
            hidden_schemas[repr(func)] = validator_name
            return func
        return decorator

    def dummy(f):
        if False:
            i = 10
            return i + 15
        return f
    return dummy

def schema_extractor_extended(func):
    if False:
        while True:
            i = 10
    if EnableSchemaExtraction:

        def decorate(*args, **kwargs):
            if False:
                return 10
            ret = func(*args, **kwargs)
            assert len(args) == 2
            extended_schemas[repr(ret)] = args
            return ret
        return decorate
    return func

def schema_extractor_list(func):
    if False:
        print('Hello World!')
    if EnableSchemaExtraction:

        def decorate(*args, **kwargs):
            if False:
                print('Hello World!')
            ret = func(*args, **kwargs)
            list_schemas[repr(ret)] = args
            return ret
        return decorate
    return func

def schema_extractor_registry(registry):
    if False:
        i = 10
        return i + 15
    if EnableSchemaExtraction:

        def decorator(func):
            if False:
                return 10
            registry_schemas[repr(func)] = registry
            return func
        return decorator

    def dummy(f):
        if False:
            i = 10
            return i + 15
        return f
    return dummy

def schema_extractor_typed(func):
    if False:
        return 10
    if EnableSchemaExtraction:

        def decorate(*args, **kwargs):
            if False:
                print('Hello World!')
            ret = func(*args, **kwargs)
            typed_schemas[repr(ret)] = (args, kwargs)
            return ret
        return decorate
    return func