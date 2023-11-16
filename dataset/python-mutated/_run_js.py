from typing import Any

def run_js(code: str, /) -> Any:
    if False:
        print('Hello World!')
    "\n    A wrapper for the JavaScript 'eval' function.\n\n    Runs 'code' as a Javascript code string and returns the result. Unlike\n    JavaScript's 'eval', if 'code' is not a string we raise a TypeError.\n    "
    from js import eval
    if not isinstance(code, str):
        raise TypeError(f"argument should have type 'string' not type '{type(code).__name__}'")
    return eval(code)