import json
import re

def parse_partial_json(s):
    if False:
        while True:
            i = 10
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    new_s = ''
    stack = []
    is_inside_string = False
    escaped = False
    for char in s:
        if is_inside_string:
            if char == '"' and (not escaped):
                is_inside_string = False
            elif char == '\n' and (not escaped):
                char = '\\n'
            elif char == '\\':
                escaped = not escaped
            else:
                escaped = False
        elif char == '"':
            is_inside_string = True
            escaped = False
        elif char == '{':
            stack.append('}')
        elif char == '[':
            stack.append(']')
        elif char == '}' or char == ']':
            if stack and stack[-1] == char:
                stack.pop()
            else:
                return None
        new_s += char
    if is_inside_string:
        new_s += '"'
    for closing_char in reversed(stack):
        new_s += closing_char
    try:
        return json.loads(new_s)
    except json.JSONDecodeError:
        return None