import typing
from vimspector.debug_session import DebugSession
from vimspector import utils
_session_manager = None

class SessionManager:
    next_session_id: int
    sessions: typing.Dict[int, DebugSession]
    api_prefix: str = ''

    def __init__(self):
        if False:
            print('Hello World!')
        self.Reset()

    def Reset(self):
        if False:
            while True:
                i = 10
        self.next_session_id = 0
        self.sessions = {}

    def NewSession(self, *args, **kwargs) -> DebugSession:
        if False:
            i = 10
            return i + 15
        session_id = self.next_session_id
        self.next_session_id += 1
        session = DebugSession(session_id, self, self.api_prefix, *args, **kwargs)
        self.sessions[session_id] = session
        return session

    def DestroySession(self, session: DebugSession):
        if False:
            print('Hello World!')
        try:
            session = self.sessions.pop(session.session_id)
        except KeyError:
            return

    def DestroyRootSession(self, session: DebugSession, active_session: DebugSession):
        if False:
            while True:
                i = 10
        if session.HasUI() or session.Connection():
            utils.UserMessage("Can't destroy active session; use VimspectorReset", error=True)
            return active_session
        try:
            self.sessions.pop(session.session_id)
            session.Destroy()
        except KeyError:
            utils.UserMessage("Session doesn't exist", error=True)
            return active_session
        if active_session != session:
            return active_session
        for existing_session in self.sessions.values():
            if not existing_session.parent_session:
                return existing_session
        return None

    def GetSession(self, session_id) -> DebugSession:
        if False:
            return 10
        return self.sessions.get(session_id)

    def GetSessionNames(self) -> typing.List[str]:
        if False:
            while True:
                i = 10
        return [s.Name() for s in self.sessions.values() if not s.parent_session and s.Name()]

    def SessionsWithInvalidUI(self):
        if False:
            print('Hello World!')
        for (_, session) in self.sessions.items():
            if not session.parent_session and (not session.HasUI()):
                yield session

    def FindSessionByTab(self, tabnr: int) -> DebugSession:
        if False:
            while True:
                i = 10
        for (_, session) in self.sessions.items():
            if session.IsUITab(tabnr):
                return session
        return None

    def FindSessionByName(self, name) -> DebugSession:
        if False:
            return 10
        for (_, session) in self.sessions.items():
            if session.Name() == name:
                return session
        return None

    def SessionForTab(self, tabnr) -> DebugSession:
        if False:
            print('Hello World!')
        session: DebugSession
        for (_, session) in self.sessions.items():
            if session.IsUITab(tabnr):
                return session
        return None

def Get():
    if False:
        i = 10
        return i + 15
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager