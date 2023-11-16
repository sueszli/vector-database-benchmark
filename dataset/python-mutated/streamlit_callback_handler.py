"""
LangChain CallbackHandler that prints to streamlit.

This is a special API that's imported and used by LangChain itself. Any updates
to the public API (the StreamlitCallbackHandler constructor, and the entirety
of LLMThoughtLabeler) *must* remain backwards-compatible to avoid breaking
LangChain.

This means that it's acceptable to add new optional kwargs to StreamlitCallbackHandler,
but no new positional args or required kwargs should be added, and no existing
args should be removed. If we need to overhaul the API, we must ensure that a
compatible API continues to exist.

Any major change to the StreamlitCallbackHandler should be tested by importing
the API *from LangChain itself*.
"""
from __future__ import annotations
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from streamlit.runtime.metrics_util import gather_metrics
if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.elements.lib.mutable_status_container import StatusContainer

def _convert_newlines(text: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Convert newline characters to markdown newline sequences\n    (space, space, newline).\n    '
    return text.replace('\n', '  \n')
MAX_TOOL_INPUT_STR_LENGTH = 60

class LLMThoughtState(Enum):
    THINKING = 'THINKING'
    RUNNING_TOOL = 'RUNNING_TOOL'
    COMPLETE = 'COMPLETE'
    ERROR = 'ERROR'

class ToolRecord(NamedTuple):
    name: str
    input_str: str

class LLMThoughtLabeler:
    """
    Generates markdown labels for LLMThought containers. Pass a custom
    subclass of this to StreamlitCallbackHandler to override its default
    labeling logic.
    """

    def get_initial_label(self) -> str:
        if False:
            print('Hello World!')
        "Return the markdown label for a new LLMThought that doesn't have\n        an associated tool yet.\n        "
        return 'Thinking...'

    def get_tool_label(self, tool: ToolRecord, is_complete: bool) -> str:
        if False:
            while True:
                i = 10
        "Return the label for an LLMThought that has an associated\n        tool.\n\n        Parameters\n        ----------\n        tool\n            The tool's ToolRecord\n\n        is_complete\n            True if the thought is complete; False if the thought\n            is still receiving input.\n\n        Returns\n        -------\n        The markdown label for the thought's container.\n\n        "
        input_str = tool.input_str
        name = tool.name
        if name == '_Exception':
            name = 'Parsing error'
        input_str_len = min(MAX_TOOL_INPUT_STR_LENGTH, len(input_str))
        input_str = input_str[:input_str_len]
        if len(tool.input_str) > input_str_len:
            input_str = input_str + '...'
        input_str = input_str.replace('\n', ' ')
        return f'**{name}:** {input_str}'

    def get_final_agent_thought_label(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the markdown label for the agent\'s final thought -\n        the "Now I have the answer" thought, that doesn\'t involve\n        a tool.\n        '
        return '**Complete!**'

class LLMThought:
    """Encapsulates the Streamlit UI for a single LLM 'thought' during a LangChain Agent
    run. Each tool usage gets its own thought; and runs also generally having a
    concluding thought where the Agent determines that it has an answer to the prompt.

    Each thought gets its own expander UI.
    """

    def __init__(self, parent_container: DeltaGenerator, labeler: LLMThoughtLabeler, expanded: bool, collapse_on_complete: bool):
        if False:
            print('Hello World!')
        self._container = parent_container.status(labeler.get_initial_label(), expanded=expanded)
        self._state = LLMThoughtState.THINKING
        self._llm_token_stream = ''
        self._llm_token_stream_placeholder: Optional[DeltaGenerator] = None
        self._last_tool: Optional[ToolRecord] = None
        self._collapse_on_complete = collapse_on_complete
        self._labeler = labeler

    @property
    def container(self) -> 'StatusContainer':
        if False:
            i = 10
            return i + 15
        "The container we're writing into."
        return self._container

    @property
    def last_tool(self) -> Optional[ToolRecord]:
        if False:
            print('Hello World!')
        'The last tool executed by this thought'
        return self._last_tool

    def _reset_llm_token_stream(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._llm_token_stream_placeholder is not None:
            self._llm_token_stream_placeholder.markdown(self._llm_token_stream)
        self._llm_token_stream = ''
        self._llm_token_stream_placeholder = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        self._reset_llm_token_stream()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self._llm_token_stream += _convert_newlines(token)
        if self._llm_token_stream_placeholder is None:
            self._llm_token_stream_placeholder = self._container.empty()
        self._llm_token_stream_placeholder.markdown(self._llm_token_stream + '▕')

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self._reset_llm_token_stream()

    def on_llm_error(self, error: BaseException, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._container.markdown('**LLM encountered an error...**')
        self._container.exception(error)
        self._state = LLMThoughtState.ERROR

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self._state = LLMThoughtState.RUNNING_TOOL
        tool_name = serialized['name']
        self._last_tool = ToolRecord(name=tool_name, input_str=input_str)
        self._container.update(label=self._labeler.get_tool_label(self._last_tool, is_complete=False), state='running')
        if len(input_str) > MAX_TOOL_INPUT_STR_LENGTH:
            self._container.markdown(f'**Input:**\n\n{input_str}\n\n**Output:**')

    def on_tool_end(self, output: str, color: Optional[str]=None, observation_prefix: Optional[str]=None, llm_prefix: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._container.markdown(output)

    def on_tool_error(self, error: BaseException, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        self._container.markdown('**Tool encountered an error...**')
        self._container.exception(error)
        self._container.update(state='error')

    def on_agent_action(self, action: AgentAction, color: Optional[str]=None, **kwargs: Any) -> Any:
        if False:
            i = 10
            return i + 15
        pass

    def complete(self, final_label: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Finish the thought.'
        if final_label is None and self._state == LLMThoughtState.RUNNING_TOOL:
            assert self._last_tool is not None, '_last_tool should never be null when _state == RUNNING_TOOL'
            final_label = self._labeler.get_tool_label(self._last_tool, is_complete=True)
        if self._last_tool and self._last_tool.name == '_Exception':
            self._state = LLMThoughtState.ERROR
        elif self._state != LLMThoughtState.ERROR:
            self._state = LLMThoughtState.COMPLETE
        if self._collapse_on_complete:
            time.sleep(0.25)
        self._container.update(label=final_label, expanded=False if self._collapse_on_complete else None, state='error' if self._state == LLMThoughtState.ERROR else 'complete')

class StreamlitCallbackHandler(BaseCallbackHandler):

    @gather_metrics('external.langchain.StreamlitCallbackHandler')
    def __init__(self, parent_container: DeltaGenerator, *, max_thought_containers: int=4, expand_new_thoughts: bool=False, collapse_completed_thoughts: bool=False, thought_labeler: Optional[LLMThoughtLabeler]=None):
        if False:
            print('Hello World!')
        'Construct a new StreamlitCallbackHandler. This CallbackHandler is geared\n        towards use with a LangChain Agent; it displays the Agent\'s LLM and tool-usage\n        "thoughts" inside a series of Streamlit expanders.\n\n        Parameters\n        ----------\n\n        parent_container\n            The `st.container` that will contain all the Streamlit elements that the\n            Handler creates.\n\n        max_thought_containers\n\n            .. note::\n                This parameter is deprecated and is ignored in the latest version of\n                the callback handler.\n\n            The max number of completed LLM thought containers to show at once. When\n            this threshold is reached, a new thought will cause the oldest thoughts to\n            be collapsed into a "History" expander. Defaults to 4.\n\n        expand_new_thoughts\n            Each LLM "thought" gets its own `st.expander`. This param controls whether\n            that expander is expanded by default. Defaults to False.\n\n        collapse_completed_thoughts\n            If True, LLM thought expanders will be collapsed when completed.\n            Defaults to False.\n\n        thought_labeler\n            An optional custom LLMThoughtLabeler instance. If unspecified, the handler\n            will use the default thought labeling logic. Defaults to None.\n        '
        self._parent_container = parent_container
        self._history_parent = parent_container.container()
        self._current_thought: Optional[LLMThought] = None
        self._completed_thoughts: List[LLMThought] = []
        self._max_thought_containers = max(max_thought_containers, 1)
        self._expand_new_thoughts = expand_new_thoughts
        self._collapse_completed_thoughts = collapse_completed_thoughts
        self._thought_labeler = thought_labeler or LLMThoughtLabeler()

    def _require_current_thought(self) -> LLMThought:
        if False:
            return 10
        'Return our current LLMThought. Raise an error if we have no current\n        thought.\n        '
        if self._current_thought is None:
            raise RuntimeError('Current LLMThought is unexpectedly None!')
        return self._current_thought

    def _get_last_completed_thought(self) -> Optional[LLMThought]:
        if False:
            print('Hello World!')
        "Return our most recent completed LLMThought, or None if we don't have one."
        if len(self._completed_thoughts) > 0:
            return self._completed_thoughts[len(self._completed_thoughts) - 1]
        return None

    def _complete_current_thought(self, final_label: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Complete the current thought, optionally assigning it a new label.\n        Add it to our _completed_thoughts list.\n        '
        thought = self._require_current_thought()
        thought.complete(final_label)
        self._completed_thoughts.append(thought)
        self._current_thought = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        if self._current_thought is None:
            self._current_thought = LLMThought(parent_container=self._parent_container, expanded=self._expand_new_thoughts, collapse_on_complete=self._collapse_completed_thoughts, labeler=self._thought_labeler)
        self._current_thought.on_llm_start(serialized, prompts)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._require_current_thought().on_llm_new_token(token, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._require_current_thought().on_llm_end(response, **kwargs)

    def on_llm_error(self, error: BaseException, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        self._require_current_thought().on_llm_error(error, **kwargs)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        self._require_current_thought().on_tool_start(serialized, input_str, **kwargs)

    def on_tool_end(self, output: str, color: Optional[str]=None, observation_prefix: Optional[str]=None, llm_prefix: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._require_current_thought().on_tool_end(output, color, observation_prefix, llm_prefix, **kwargs)
        self._complete_current_thought()

    def on_tool_error(self, error: BaseException, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._require_current_thought().on_tool_error(error, **kwargs)

    def on_agent_action(self, action: AgentAction, color: Optional[str]=None, **kwargs: Any) -> Any:
        if False:
            while True:
                i = 10
        self._require_current_thought().on_agent_action(action, color, **kwargs)

    def on_agent_finish(self, finish: AgentFinish, color: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        if self._current_thought is not None:
            self._current_thought.complete(self._thought_labeler.get_final_agent_thought_label())
            self._current_thought = None