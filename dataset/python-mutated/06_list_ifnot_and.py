def list_public_methods(obj):
    if False:
        print('Hello World!')
    return [member for member in dir(obj) if not member.startswith('_') and hasattr(getattr(obj, member), '__call__')]