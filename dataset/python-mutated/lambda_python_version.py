import sys

def handler(event, context):
    if False:
        print('Hello World!')
    return {'version': 'python{major}.{minor}'.format(major=sys.version_info.major, minor=sys.version_info.minor)}