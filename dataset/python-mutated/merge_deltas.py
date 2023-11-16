import json
import re

def merge_deltas(original, delta):
    if False:
        while True:
            i = 10
    '\n    Pushes the delta into the original and returns that.\n\n    Great for reconstructing OpenAI streaming responses -> complete message objects.\n    '
    for (key, value) in delta.items():
        if isinstance(value, dict):
            if key not in original:
                original[key] = value
            else:
                merge_deltas(original[key], value)
        elif key in original:
            original[key] += value
        else:
            original[key] = value
    return original