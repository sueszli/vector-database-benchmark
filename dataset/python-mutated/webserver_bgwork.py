from typing import Any, Dict, Literal, Optional, TypedDict
from uuid import uuid4
from freqtrade.exchange.exchange import Exchange

class JobsContainer(TypedDict):
    category: Literal['pairlist']
    is_running: bool
    status: str
    progress: Optional[float]
    result: Any
    error: Optional[str]

class ApiBG:
    bt: Dict[str, Any] = {'bt': None, 'data': None, 'timerange': None, 'last_config': {}, 'bt_error': None}
    bgtask_running: bool = False
    exchanges: Dict[str, Exchange] = {}
    jobs: Dict[str, JobsContainer] = {}
    pairlist_running: bool = False

    @staticmethod
    def get_job_id() -> str:
        if False:
            print('Hello World!')
        return str(uuid4())