from vimspector.debug_session import DebugSession
import copy

class JsDebug(object):
    parent: DebugSession

    def __init__(self, debug_session: DebugSession):
        if False:
            return 10
        self.parent = debug_session

    def OnRequest_startDebugging(self, message):
        if False:
            print('Hello World!')
        adapter = copy.deepcopy(self.parent._adapter)
        adapter.pop('command', None)
        self.parent._DoStartDebuggingRequest(message, message['arguments']['request'], message['arguments']['configuration'], adapter)
        return True