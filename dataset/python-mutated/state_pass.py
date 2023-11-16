from typing import Optional
from localstack.aws.api.stepfunctions import HistoryEventExecutionDataDetails, HistoryEventType, StateEnteredEventDetails, StateExitedEventDetails
from localstack.services.stepfunctions.asl.component.common.parameters import Parameters
from localstack.services.stepfunctions.asl.component.common.path.result_path import ResultPath
from localstack.services.stepfunctions.asl.component.state.state import CommonStateField
from localstack.services.stepfunctions.asl.component.state.state_pass.result import Result
from localstack.services.stepfunctions.asl.component.state.state_props import StateProps
from localstack.services.stepfunctions.asl.eval.environment import Environment
from localstack.services.stepfunctions.asl.utils.encoding import to_json_str

class StatePass(CommonStateField):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(StatePass, self).__init__(state_entered_event_type=HistoryEventType.PassStateEntered, state_exited_event_type=HistoryEventType.PassStateExited)
        self.result: Optional[Result] = None
        self.result_path: Optional[ResultPath] = None
        self.parameters: Optional[Parameters] = None

    def from_state_props(self, state_props: StateProps) -> None:
        if False:
            while True:
                i = 10
        super(StatePass, self).from_state_props(state_props)
        self.result = state_props.get(Result)
        self.result_path = state_props.get(ResultPath)
        self.parameters = state_props.get(Parameters)
        if self.result_path is None:
            self.result_path = ResultPath(result_path_src=ResultPath.DEFAULT_PATH)

    def _get_state_entered_event_details(self, env: Environment) -> StateEnteredEventDetails:
        if False:
            return 10
        return StateEnteredEventDetails(name=self.name, input=to_json_str(env.inp, separators=(',', ':')), inputDetails=HistoryEventExecutionDataDetails(truncated=False))

    def _get_state_exited_event_details(self, env: Environment) -> StateExitedEventDetails:
        if False:
            for i in range(10):
                print('nop')
        return StateExitedEventDetails(name=self.name, output=to_json_str(env.inp, separators=(',', ':')), outputDetails=HistoryEventExecutionDataDetails(truncated=False))

    def _eval_state(self, env: Environment) -> None:
        if False:
            print('Hello World!')
        if self.parameters:
            self.parameters.eval(env=env)
        if self.result:
            env.stack.append(self.result.result_obj)
        if self.result_path:
            self.result_path.eval(env)