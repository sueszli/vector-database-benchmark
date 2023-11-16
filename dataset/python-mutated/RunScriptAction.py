def RunScriptAction(script: str, args=''):
    if False:
        return 10
    return {'type': 'action:legacy_run_script', 'data': [script, args]}