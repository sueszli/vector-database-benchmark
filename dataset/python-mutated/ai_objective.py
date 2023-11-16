import functools
from enum import Enum, auto
from typing import Any, Callable, Generic, Optional, TypeVar
from pydantic import BaseModel, Field
from rich.prompt import Prompt
from typing_extensions import ParamSpec
from marvin.beta.assistants import Assistant, Run, Thread
from marvin.beta.assistants.formatting import pprint_message, pprint_messages
from marvin.beta.assistants.runs import CancelRun
from marvin.serializers import create_tool_from_type
from marvin.tools.assistants import AssistantTools
from marvin.utilities.asyncio import run_sync
from marvin.utilities.jinja import Environment as JinjaEnvironment
from marvin.utilities.tools import tool_from_function
T = TypeVar('T', bound=BaseModel)
P = ParamSpec('P')
INSTRUCTIONS = '\n# Current task\n\nYou are currently working on the "{{ name }}" task. Complete the task in\naccordance with the description below.\n\n# Task description\n\n{{ instructions }}\n\n{% if first_message -%}\n# No user input\n\nIf there is no user input, that\'s because the user doesn\'t know about the task\nyet. You should send a message to the user to get them to respond.\n\n{% endif %}\n\n# Completing the task\n\nAfter achieving your goal, you MUST call the `task_completed` tool to mark the\ntask as complete and move on to the next one. The payload to `task_completed` is\nwhatever information represents the task objective. For example, if your task is\nto learn a user\'s name, you should respond with their properly formatted name\nonly.\n\nYou may be expected to return a\nspecific data payload at the end of your task, which will be the input to\n`task_completed`. Note that if your instructions are to talk to the user, then\nyou must do so by creating messages, as the user can not see the `task_completed`\ntool result.\n\nDo not call `task_completed` unless you actually have the information you need.\nThe user CAN NOT see what you post to `task_completed`. It is not a way to\ncommunicate with the user.\n\n# Failing the task\n\nIt may take you a few tries to complete the task. However, if you are ultimately\nunable to work with the user to complete it, call the `task_failed` tool to mark\nthe task as failed and move on to the next one. The payload to `task_failed` is\na string describing why the task failed.\n\n{% if args or kwargs -%}\n# Task inputs\n\nIn addition to the thread messages, the following parameters were provided:\n{% set sig = inspect.signature(func) -%}\n\n{% set binds = sig.bind(*args, **kwargs) -%}\n\n{% set defaults = binds.apply_defaults() -%}\n\n{% set params = binds.arguments -%}\n\n{%for (arg, value) in params.items()-%}\n\n- {{ arg }}: {{ value }}\n\n{% endfor %}\n\n{% endif %}\n'

class ObjectiveStatus(Enum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()

class AIObjective(BaseModel, Generic[P, T]):
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    fn: Callable[P, Any]
    name: str = Field(None, description='The name of the objective')
    instructions: str = Field(None, description='The instructions for the objective')
    assistant: Optional[Assistant] = None
    tools: list[AssistantTools] = []
    max_run_iterations: int = 15
    result: Optional[T] = None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if False:
            i = 10
            return i + 15
        return run_sync(self.call(*args, **kwargs))

    async def call(self, *args, _thread_id: str=None, **kwargs):
        thread = Thread(id=_thread_id)
        if _thread_id is None:
            thread.create()
        iterations = 0
        self.status = ObjectiveStatus.IN_PROGRESS
        with Assistant() as assistant:
            while self.status == ObjectiveStatus.IN_PROGRESS:
                iterations += 1
                if iterations > self.max_run_iterations:
                    raise ValueError('Max run iterations exceeded')
                instructions = self.get_instructions(*args, iterations=iterations, **kwargs)
                if iterations > 1:
                    user_input = Prompt.ask('Your message')
                    msg = thread.add(user_input)
                    pprint_message(msg)
                else:
                    msg = None
                run = Run(assistant=assistant, thread=thread, additional_instructions=instructions, additional_tools=[self._task_completed_tool, self._task_failed_tool])
                await run.run_async()
                messages = thread.get_messages(after_message=msg.id if msg else None)
                pprint_messages(messages)
            if self.status == ObjectiveStatus.FAILED:
                raise ValueError(f'Objective failed: {self.result}')
        return self.result

    def get_instructions(self, iterations: int, *args: P.args, **kwargs: P.kwargs) -> str:
        if False:
            for i in range(10):
                print('nop')
        return JinjaEnvironment.render(INSTRUCTIONS, first_message=iterations == 1, name=self.name, instructions=self.instructions, func=self.fn, args=args, kwargs=kwargs)

    @property
    def _task_completed_tool(self):
        if False:
            i = 10
            return i + 15
        tool = create_tool_from_type(_type=self.fn.__annotations__['return'], model_name='task_completed', model_description='Use this tool to complete the objective and provide a result that contains its result.', field_name='result', field_description='The objective result')

        def task_completed(result: T):
            if False:
                for i in range(10):
                    print('nop')
            self.status = ObjectiveStatus.COMPLETED
            self.result = result
            return 'The task is complete. Do NOT continue talking at this time.'
        tool.function.python_fn = task_completed
        return tool

    @property
    def _task_failed_tool(self):
        if False:
            print('Hello World!')

        def task_failed(reason: str) -> None:
            if False:
                print('Hello World!')
            'Indicate that the task failed for the provided `reason`.'
            self.status = ObjectiveStatus.FAILED
            self.result = reason
            raise CancelRun()
        return tool_from_function(task_failed)

def ai_objective(*args, name=None, instructions=None, tools: list[AssistantTools]=None):
    if False:
        print('Hello World!')

    def decorator(func):
        if False:
            return 10

        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            if False:
                print('Hello World!')
            ai_objective_instance = AIObjective(fn=func, name=name or func.__name__, instructions=instructions or func.__doc__, tools=tools or [])
            return ai_objective_instance(*func_args, **func_kwargs)
        return wrapper
    if args and callable(args[0]):
        return decorator(args[0])
    return decorator