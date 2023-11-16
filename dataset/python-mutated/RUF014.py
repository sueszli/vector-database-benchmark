def after_return():
    if False:
        for i in range(10):
            print('nop')
    return 'reachable'
    return 'unreachable'

async def also_works_on_async_functions():
    return 'reachable'
    return 'unreachable'

def if_always_true():
    if False:
        print('Hello World!')
    if True:
        return 'reachable'
    return 'unreachable'

def if_always_false():
    if False:
        print('Hello World!')
    if False:
        return 'unreachable'
    return 'reachable'

def if_elif_always_false():
    if False:
        while True:
            i = 10
    if False:
        return 'unreachable'
    elif False:
        return 'also unreachable'
    return 'reachable'

def if_elif_always_true():
    if False:
        return 10
    if False:
        return 'unreachable'
    elif True:
        return 'reachable'
    return 'also unreachable'

def ends_with_if():
    if False:
        while True:
            i = 10
    if False:
        return 'unreachable'
    else:
        return 'reachable'

def infinite_loop():
    if False:
        return 10
    while True:
        continue
    return 'unreachable'
'  TODO: we could determine these, but we don\'t yet.\ndef for_range_return():\n    for i in range(10):\n        if i == 5:\n            return "reachable"\n    return "unreachable"\n\ndef for_range_else():\n    for i in range(111):\n        if i == 5:\n            return "reachable"\n    else:\n        return "unreachable"\n    return "also unreachable"\n\ndef for_range_break():\n    for i in range(13):\n        return "reachable"\n    return "unreachable"\n\ndef for_range_if_break():\n    for i in range(1110):\n        if True:\n            return "reachable"\n    return "unreachable"\n'

def match_wildcard(status):
    if False:
        return 10
    match status:
        case _:
            return 'reachable'
    return 'unreachable'

def match_case_and_wildcard(status):
    if False:
        i = 10
        return i + 15
    match status:
        case 1:
            return 'reachable'
        case _:
            return 'reachable'
    return 'unreachable'

def raise_exception():
    if False:
        while True:
            i = 10
    raise Exception
    return 'unreachable'

def while_false():
    if False:
        return 10
    while False:
        return 'unreachable'
    return 'reachable'

def while_false_else():
    if False:
        return 10
    while False:
        return 'unreachable'
    else:
        return 'reachable'

def while_false_else_return():
    if False:
        print('Hello World!')
    while False:
        return 'unreachable'
    else:
        return 'reachable'
    return 'also unreachable'

def while_true():
    if False:
        for i in range(10):
            print('nop')
    while True:
        return 'reachable'
    return 'unreachable'

def while_true_else():
    if False:
        while True:
            i = 10
    while True:
        return 'reachable'
    else:
        return 'unreachable'

def while_true_else_return():
    if False:
        return 10
    while True:
        return 'reachable'
    else:
        return 'unreachable'
    return 'also unreachable'

def while_false_var_i():
    if False:
        i = 10
        return i + 15
    i = 0
    while False:
        i += 1
    return i

def while_true_var_i():
    if False:
        print('Hello World!')
    i = 0
    while True:
        i += 1
    return i

def while_infinite():
    if False:
        i = 10
        return i + 15
    while True:
        pass
    return 'unreachable'

def while_if_true():
    if False:
        i = 10
        return i + 15
    while True:
        if True:
            return 'reachable'
    return 'unreachable'

def bokeh1(self, obj: BytesRep) -> bytes:
    if False:
        i = 10
        return i + 15
    data = obj['data']
    if isinstance(data, str):
        return base64.b64decode(data)
    elif isinstance(data, Buffer):
        buffer = data
    else:
        id = data['id']
        if id in self._buffers:
            buffer = self._buffers[id]
        else:
            self.error(f"can't resolve buffer '{id}'")
    return buffer.data
'\nTODO: because `try` statements aren\'t handled this triggers a false positive as\nthe last statement is reached, but the rules thinks it isn\'t (it doesn\'t\nsee/process the break statement).\n\n# Test case found in the Bokeh repository that trigger a false positive.\ndef bokeh2(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:\n    self.stop_serving = False\n    while True:\n        try:\n            self.server = HTTPServer((host, port), HtmlOnlyHandler)\n            self.host = host\n            self.port = port\n            break\n        except OSError:\n            log.debug(f"port {port} is in use, trying to next one")\n            port += 1\n\n    self.thread = threading.Thread(target=self._run_web_server)\n'