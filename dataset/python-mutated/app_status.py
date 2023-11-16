from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel

class WorkStatus(BaseModel):
    """The ``WorkStatus`` captures the status of a work according to the app."""
    stage: str
    timestamp: float
    reason: Optional[str] = None
    message: Optional[str] = None
    count: int = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        assert self.timestamp > 0
        assert self.timestamp < int(datetime.now().timestamp()) + 10

class AppStatus(BaseModel):
    """The ``AppStatus`` captures the current status of the app and its components."""
    is_ui_ready: bool
    work_statuses: Dict[str, WorkStatus]