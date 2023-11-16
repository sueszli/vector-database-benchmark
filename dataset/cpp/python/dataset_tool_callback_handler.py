import json
import logging
from json import JSONDecodeError

from typing import Any, Dict, List, Union, Optional

from langchain.callbacks.base import BaseCallbackHandler

from core.callback_handler.entity.dataset_query import DatasetQueryObj
from core.conversation_message_task import ConversationMessageTask


class DatasetToolCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to std out."""
    raise_error: bool = True

    def __init__(self, conversation_message_task: ConversationMessageTask) -> None:
        """Initialize callback handler."""
        self.queries = []
        self.conversation_message_task = conversation_message_task

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return True

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return True

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        tool_name: str = serialized.get('name')
        dataset_id = tool_name.removeprefix('dataset-')

        try:
            input_dict = json.loads(input_str.replace("'", "\""))
            query = input_dict.get('query')
        except JSONDecodeError:
            query = input_str

        self.conversation_message_task.on_dataset_query_end(DatasetQueryObj(dataset_id=dataset_id, query=query))

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pass


    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        logging.debug("Dataset tool on_llm_error: %s", error)
