from flaml.autogen.agentchat.agent import Agent
from flaml.autogen.agentchat.assistant_agent import AssistantAgent
from typing import Callable, Dict, Optional, Union, List, Tuple, Any

class RetrieveAssistantAgent(AssistantAgent):
    """(Experimental) Retrieve Assistant agent, designed to solve a task with LLM.

    RetrieveAssistantAgent is a subclass of AssistantAgent configured with a default system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    `human_input_mode` is default to "NEVER"
    and `code_execution_config` is default to False.
    This agent doesn't execute code by default, and expects the user to execute the code.
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.register_reply(Agent, RetrieveAssistantAgent._generate_retrieve_assistant_reply)

    def _generate_retrieve_assistant_reply(self, messages: Optional[List[Dict]]=None, sender: Optional[Agent]=None, config: Optional[Any]=None) -> Tuple[bool, Union[str, Dict, None]]:
        if False:
            i = 10
            return i + 15
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if 'exitcode: 0 (execution succeeded)' in message.get('content', ''):
            return (True, 'TERMINATE')
        elif 'UPDATE CONTEXT' in message.get('content', '')[-20:].upper() or 'UPDATE CONTEXT' in message.get('content', '')[:20].upper():
            return (True, 'UPDATE CONTEXT')
        else:
            return (False, None)