from functools import lru_cache
from typing import Any, Dict, List
import yaml
from opensfm import context
from opensfm import io

@lru_cache(1)
def sensor_data() -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    with io.open_rt(context.SENSOR_DATA) as f:
        data = io.json_load(f)
    return {k.lower(): v for (k, v) in data.items()}

@lru_cache(1)
def camera_calibration() -> List[Dict[str, Any]]:
    if False:
        return 10
    with io.open_rt(context.CAMERA_CALIBRATION) as f:
        data = yaml.safe_load(f)
    return data