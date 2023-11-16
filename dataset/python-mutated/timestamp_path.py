import datetime
from typing import Final
from localstack.services.stepfunctions.asl.component.state.state_wait.wait_function.timestamp import Timestamp
from localstack.services.stepfunctions.asl.component.state.state_wait.wait_function.wait_function import WaitFunction
from localstack.services.stepfunctions.asl.eval.environment import Environment
from localstack.services.stepfunctions.asl.utils.json_path import JSONPathUtils

class TimestampPath(WaitFunction):

    def __init__(self, path: str):
        if False:
            while True:
                i = 10
        self.path: Final[str] = path

    def _get_wait_seconds(self, env: Environment) -> int:
        if False:
            return 10
        timestamp_str = JSONPathUtils.extract_json(self.path, env.inp)
        timestamp = datetime.datetime.strptime(timestamp_str, Timestamp.TIMESTAMP_FORMAT)
        delta = timestamp - datetime.datetime.today()
        delta_sec = int(delta.total_seconds())
        return delta_sec