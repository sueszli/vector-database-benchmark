__revision__ = 'src/engine/SCons/Tool/mssdk.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
"engine.SCons.Tool.mssdk\n\nTool-specific initialization for Microsoft SDKs, both Platform\nSDKs and Windows SDKs.\n\nThere normally shouldn't be any need to import this module directly.\nIt will usually be imported through the generic SCons.Tool.Tool()\nselection method.\n"
from .MSCommon import mssdk_exists, mssdk_setup_env

def generate(env):
    if False:
        return 10
    'Add construction variables for an MS SDK to an Environment.'
    mssdk_setup_env(env)

def exists(env):
    if False:
        print('Hello World!')
    return mssdk_exists()