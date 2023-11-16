import re
import ast
import json

def _analyze_ast(contents):
    if False:
        while True:
            i = 10
    try:
        return ast.literal_eval(contents)
    except SyntaxError:
        pass
    try:
        contents = re.sub(re.compile('/\\*.*?\\*/', re.DOTALL), '', contents)
        contents = re.sub(re.compile('#.*?\\n'), '', contents)
        match = re.match('^([^{]+)', contents)
        if match:
            contents = contents.replace(match.group(1), '')
        return ast.literal_eval(contents)
    except SyntaxError:
        pass
    return False

def _analyze_manual(contents):
    if False:
        return 10
    capabilities = {}
    code_lines = contents.split('\n')
    for line in code_lines:
        if 'desired_cap = {' in line:
            line = line.split('desired_cap = {')[1]
        data = re.match("^\\s*'([\\S\\s]+)'\\s*:\\s*'([\\S\\s]+)'\\s*[,}]?\\s*$", line)
        if data:
            key = data.group(1)
            value = data.group(2)
            capabilities[key] = value
            continue
        data = re.match('^\\s*"([\\S\\s]+)"\\s*:\\s*"([\\S\\s]+)"\\s*[,}]?\\s*$', line)
        if data:
            key = data.group(1)
            value = data.group(2)
            capabilities[key] = value
            continue
        data = re.match('^\\s*\'([\\S\\s]+)\'\\s*:\\s*"([\\S\\s]+)"\\s*[,}]?\\s*$', line)
        if data:
            key = data.group(1)
            value = data.group(2)
            capabilities[key] = value
            continue
        data = re.match('^\\s*"([\\S\\s]+)"\\s*:\\s*\'([\\S\\s]+)\'\\s*[,}]?\\s*$', line)
        if data:
            key = data.group(1)
            value = data.group(2)
            capabilities[key] = value
            continue
        data = re.match('^\\s*"([\\S\\s]+)"\\s*:\\s*True\\s*[,}]?\\s*$', line)
        if data:
            key = data.group(1)
            value = True
            capabilities[key] = value
            continue
        data = re.match("^\\s*'([\\S\\s]+)'\\s*:\\s*True\\s*[,}]?\\s*$", line)
        if data:
            key = data.group(1)
            value = True
            capabilities[key] = value
            continue
        data = re.match('^\\s*"([\\S\\s]+)"\\s*:\\s*False\\s*[,}]?\\s*$', line)
        if data:
            key = data.group(1)
            value = False
            capabilities[key] = value
            continue
        data = re.match("^\\s*'([\\S\\s]+)'\\s*:\\s*False\\s*[,}]?\\s*$", line)
        if data:
            key = data.group(1)
            value = False
            capabilities[key] = value
            continue
        data = re.match("^\\s*caps\\['([\\S\\s]+)'\\]\\s*=\\s*'([\\S\\s]+)'\\s*$", line)
        if data:
            key = data.group(1)
            value = data.group(2)
            capabilities[key] = value
            continue
        data = re.match('^\\s*caps\\["([\\S\\s]+)"\\]\\s*=\\s*"([\\S\\s]+)"\\s*$', line)
        if data:
            key = data.group(1)
            value = data.group(2)
            capabilities[key] = value
            continue
        data = re.match('^\\s*caps\\[\'([\\S\\s]+)\'\\]\\s*=\\s*"([\\S\\s]+)"\\s*$', line)
        if data:
            key = data.group(1)
            value = data.group(2)
            capabilities[key] = value
            continue
        data = re.match('^\\s*caps\\["([\\S\\s]+)"\\]\\s*=\\s*\'([\\S\\s]+)\'\\s*$', line)
        if data:
            key = data.group(1)
            value = data.group(2)
            capabilities[key] = value
            continue
        data = re.match('^\\s*caps\\["([\\S\\s]+)"\\]\\s*=\\s*True\\s*$', line)
        if data:
            key = data.group(1)
            value = True
            capabilities[key] = value
            continue
        data = re.match("^\\s*caps\\['([\\S\\s]+)'\\]\\s*=\\s*True\\s*$", line)
        if data:
            key = data.group(1)
            value = True
            capabilities[key] = value
            continue
        data = re.match('^\\s*caps\\["([\\S\\s]+)"\\]\\s*=\\s*False\\s*$', line)
        if data:
            key = data.group(1)
            value = False
            capabilities[key] = value
            continue
        data = re.match("^\\s*caps\\['([\\S\\s]+)'\\]\\s*=\\s*False\\s*$", line)
        if data:
            key = data.group(1)
            value = False
            capabilities[key] = value
            continue
    return capabilities

def _read_file(file):
    if False:
        for i in range(10):
            print('nop')
    f = open(file, 'r')
    data = f.read()
    f.close()
    return data

def _parse_py_file(cap_file):
    if False:
        while True:
            i = 10
    all_code = _read_file(cap_file)
    capabilities = _analyze_ast(all_code)
    if not capabilities:
        capabilities = _analyze_manual(all_code)
    return capabilities

def _parse_json_file(cap_file):
    if False:
        print('Hello World!')
    all_code = _read_file(cap_file)
    return json.loads(all_code)

def get_desired_capabilities(cap_file):
    if False:
        for i in range(10):
            print('nop')
    if cap_file.endswith('.py'):
        capabilities = _parse_py_file(cap_file)
    elif cap_file.endswith('.json'):
        capabilities = _parse_json_file(cap_file)
    else:
        raise Exception('\n\n`%s` is not a Python or JSON file!\n' % cap_file)
    if len(capabilities.keys()) == 0:
        raise Exception('Unable to parse desired capabilities file!')
    return capabilities