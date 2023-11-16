import json

def to_json(out, err, code):
    if False:
        while True:
            i = 10
    payload = {}
    if err:
        payload['err'] = err
        payload['exit_code'] = code
        return json.dumps(payload)
    payload['pkg_info'] = out
    payload['exit_code'] = code
    return json.dumps(payload)