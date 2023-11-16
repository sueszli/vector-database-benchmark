import torch._C._lazy

def dump(dot_file_name: str):
    if False:
        i = 10
        return i + 15
    'Dump TrieCache in the dot format'
    return torch._C._lazy._dump_ir_cache(dot_file_name)

def reset():
    if False:
        print('Hello World!')
    'Clear TrieCache. This is needed in testing to avoid\n    node reusing between different tests.\n    '
    return torch._C._lazy._clear_ir_cache()