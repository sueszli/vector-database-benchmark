from localstack.aws.api.stepfunctions import HistoryEventType
from localstack.services.stepfunctions.asl.component.state.state_execution.execute_state import ExecutionState
from localstack.services.stepfunctions.asl.component.state.state_execution.state_parallel.branches_decl import BranchesDecl
from localstack.services.stepfunctions.asl.component.state.state_props import StateProps
from localstack.services.stepfunctions.asl.eval.environment import Environment

class StateParallel(ExecutionState):
    branches: BranchesDecl

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(state_entered_event_type=HistoryEventType.ParallelStateEntered, state_exited_event_type=HistoryEventType.ParallelStateExited)

    def from_state_props(self, state_props: StateProps) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(StateParallel, self).from_state_props(state_props)
        self.branches = state_props.get(typ=BranchesDecl, raise_on_missing=ValueError(f"Missing Branches definition in props '{state_props}'."))

    def _eval_execution(self, env: Environment) -> None:
        if False:
            i = 10
            return i + 15
        env.event_history.add_event(context=env.event_history_context, hist_type_event=HistoryEventType.ParallelStateStarted)
        self.branches.eval(env)
        env.event_history.add_event(context=env.event_history_context, hist_type_event=HistoryEventType.ParallelStateSucceeded, update_source_event_id=False)