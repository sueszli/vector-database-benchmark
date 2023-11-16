from typing import List, MutableMapping, Optional
from cachetools import TTLCache
from streamlit.runtime.session_manager import SessionInfo, SessionStorage

class MemorySessionStorage(SessionStorage):
    """A SessionStorage that stores sessions in memory.

    At most maxsize sessions are stored with a TTL of ttl seconds. This class is really
    just a thin wrapper around cachetools.TTLCache that complies with the SessionStorage
    protocol.
    """

    def __init__(self, maxsize: int=128, ttl_seconds: int=2 * 60) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Instantiate a new MemorySessionStorage.\n\n        Parameters\n        ----------\n        maxsize\n            The maximum number of sessions we allow to be stored in this\n            MemorySessionStorage. If an entry needs to be removed because we have\n            exceeded this number, either\n              * an expired entry is removed, or\n              * the least recently used entry is removed (if no entries have expired).\n\n        ttl_seconds\n            The time in seconds for an entry added to a MemorySessionStorage to live.\n            After this amount of time has passed for a given entry, it becomes\n            inaccessible and will be removed eventually.\n        '
        self._cache: MutableMapping[str, SessionInfo] = TTLCache(maxsize=maxsize, ttl=ttl_seconds)

    def get(self, session_id: str) -> Optional[SessionInfo]:
        if False:
            while True:
                i = 10
        return self._cache.get(session_id, None)

    def save(self, session_info: SessionInfo) -> None:
        if False:
            return 10
        self._cache[session_info.session.id] = session_info

    def delete(self, session_id: str) -> None:
        if False:
            return 10
        del self._cache[session_id]

    def list(self) -> List[SessionInfo]:
        if False:
            print('Hello World!')
        return list(self._cache.values())