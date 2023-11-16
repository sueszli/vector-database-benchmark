"""
Lambda Builders-speicific utils
"""

def patch_runtime(runtime: str) -> str:
    if False:
        while True:
            i = 10
    if runtime.startswith('provided'):
        runtime = 'provided'
    return runtime.replace('.al2', '')