import inspect

def trim_docstring(docstring):
    if False:
        i = 10
        return i + 15
    return inspect.cleandoc(docstring) if docstring else None