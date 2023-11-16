from __future__ import annotations

def is_quoted(data):
    if False:
        while True:
            i = 10
    return len(data) > 1 and data[0] == data[-1] and (data[0] in ('"', "'")) and (data[-2] != '\\')

def unquote(data):
    if False:
        return 10
    ' removes first and last quotes from a string, if the string starts and ends with the same quotes '
    if is_quoted(data):
        return data[1:-1]
    return data