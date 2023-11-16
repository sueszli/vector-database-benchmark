import os
from devtools_testutils import is_live

def get_connection_str():
    if False:
        return 10
    if not is_live():
        return 'endpoint=https://sanitized.communication.azure.com/;accesskey=fake==='
    return os.getenv('COMMUNICATION_LIVETEST_DYNAMIC_CONNECTION_STRING') or os.getenv('COMMUNICATION_LIVETEST_STATIC_CONNECTION_STRING')