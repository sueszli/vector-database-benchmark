from typing import Any, Final, Optional, TypedDict
from localstack.utils.strings import long_uid

class Execution(TypedDict):
    Id: str
    Input: Optional[dict]
    Name: str
    RoleArn: str
    StartTime: str

class State(TypedDict):
    EnteredTime: str
    Name: str
    RetryCount: int

class StateMachine(TypedDict):
    Id: str
    Name: str

class Task(TypedDict):
    Token: str

class Item(TypedDict):
    Index: int
    Value: Optional[Any]

class Map(TypedDict):
    Item: Item

class ContextObject(TypedDict):
    Execution: Execution
    State: Optional[State]
    StateMachine: StateMachine
    Task: Optional[Task]
    Map: Optional[Map]

class ContextObjectManager:
    context_object: Final[ContextObject]

    def __init__(self, context_object: ContextObject):
        if False:
            while True:
                i = 10
        self.context_object = context_object

    def update_task_token(self) -> str:
        if False:
            print('Hello World!')
        new_token = long_uid()
        self.context_object['Task'] = Task(Token=new_token)
        return new_token

class ContextObjectInitData(TypedDict):
    Execution: Execution
    StateMachine: StateMachine