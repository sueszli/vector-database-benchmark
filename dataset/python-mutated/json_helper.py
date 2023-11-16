import json

def merge_dict(base, delta):
    if False:
        return 10
    '\n        Recursively merging configuration dictionaries.\n\n        Args:\n            base:  Target for merge\n            delta: Dictionary to merge into base\n    '
    for (k, dv) in delta.items():
        bv = base.get(k)
        if isinstance(dv, dict) and isinstance(bv, dict):
            merge_dict(bv, dv)
        else:
            base[k] = dv

def load_commented_json(filename):
    if False:
        print('Hello World!')
    ' Loads an JSON file, ignoring comments\n\n    Supports a trivial extension to the JSON file format.  Allow comments\n    to be embedded within the JSON, requiring that a comment be on an\n    independent line starting with \'//\' or \'#\'.\n\n    NOTE: A file created with these style comments will break strict JSON\n          parsers.  This is similar to but lighter-weight than "human json"\n          proposed at https://hjson.org\n\n    Args:\n        filename (str):  path to the commented JSON file\n\n    Returns:\n        obj: decoded Python object\n    '
    with open(filename) as f:
        contents = f.read()
    return json.loads(uncomment_json(contents))

def uncomment_json(commented_json_str):
    if False:
        print('Hello World!')
    " Removes comments from a JSON string.\n\n    Supporting a trivial extension to the JSON format.  Allow comments\n    to be embedded within the JSON, requiring that a comment be on an\n    independent line starting with '//' or '#'.\n\n    Example...\n       {\n         // comment\n         'name' : 'value'\n       }\n\n    Args:\n        commented_json_str (str):  a JSON string\n\n    Returns:\n        str: uncommented, legal JSON\n    "
    lines = commented_json_str.splitlines()
    nocomment = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('//') or stripped.startswith('#'):
            continue
        nocomment.append(line)
    return ' '.join(nocomment)