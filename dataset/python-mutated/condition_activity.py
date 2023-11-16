from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict
FREQUENCY_CONDITION_BUCKET_SIZE = timedelta(minutes=5)

class ConditionActivityType(Enum):
    CREATE_ISSUE = 0
    REGRESSION = 1
    REAPPEARED = 2
    FREQUENCY_CONDITION = 3

@dataclass
class ConditionActivity:
    group_id: int
    type: ConditionActivityType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)

def round_to_five_minute(time: datetime) -> datetime:
    if False:
        return 10
    return time - timedelta(minutes=time.minute % 5, seconds=time.second, microseconds=time.microsecond)