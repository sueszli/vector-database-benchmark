import re
from typing import List, Tuple, Any, Union, Sequence, Optional, cast
from langchain import BasePromptTemplate, PromptTemplate
from langchain.agents import StructuredChatAgent, AgentOutputParser, Agent
from langchain.agents.structured_chat.base import HUMAN_MESSAGE_TEMPLATE
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.tools import BaseTool
from langchain.agents.structured_chat.prompt import PREFIX, SUFFIX
from core.chain.llm_chain import LLMChain
from core.model_providers.models.entity.model_params import ModelMode
from core.model_providers.models.llm.base import BaseLLM
from core.tool.dataset_retriever_tool import DatasetRetrieverTool
FORMAT_INSTRUCTIONS = 'Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\nThe nouns in the format of "Thought", "Action", "Action Input", "Final Answer" must be expressed in English.\nValid "action" values: "Final Answer" or {tool_names}\n\nProvide only ONE action per $JSON_BLOB, as shown:\n\n```\n{{{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}}}\n```\n\nFollow this format:\n\nQuestion: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n... (repeat Thought/Action/Observation N times)\nThought: I know what to respond\nAction:\n```\n{{{{\n  "action": "Final Answer",\n  "action_input": "Final response to human"\n}}}}\n```'

class StructuredMultiDatasetRouterAgent(StructuredChatAgent):
    dataset_tools: Sequence[BaseTool]

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def should_use_agent(self, query: str):
        if False:
            while True:
                i = 10
        "\n        return should use agent\n        Using the ReACT mode to determine whether an agent is needed is costly,\n        so it's better to just use an Agent for reasoning, which is cheaper.\n\n        :param query:\n        :return:\n        "
        return True

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], callbacks: Callbacks=None, **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        if False:
            i = 10
            return i + 15
        'Given input, decided what to do.\n\n        Args:\n            intermediate_steps: Steps the LLM has taken to date,\n                along with observations\n            callbacks: Callbacks to run.\n            **kwargs: User inputs.\n\n        Returns:\n            Action specifying what tool to use.\n        '
        if len(self.dataset_tools) == 0:
            return AgentFinish(return_values={'output': ''}, log='')
        elif len(self.dataset_tools) == 1:
            tool = next(iter(self.dataset_tools))
            tool = cast(DatasetRetrieverTool, tool)
            rst = tool.run(tool_input={'query': kwargs['input']})
            return AgentFinish(return_values={'output': rst}, log=rst)
        if intermediate_steps:
            (_, observation) = intermediate_steps[-1]
            return AgentFinish(return_values={'output': observation}, log=observation)
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        try:
            full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        except Exception as e:
            new_exception = self.llm_chain.model_instance.handle_exceptions(e)
            raise new_exception
        try:
            agent_decision = self.output_parser.parse(full_output)
            if isinstance(agent_decision, AgentAction):
                tool_inputs = agent_decision.tool_input
                if isinstance(tool_inputs, dict) and 'query' in tool_inputs:
                    tool_inputs['query'] = kwargs['input']
                    agent_decision.tool_input = tool_inputs
                elif isinstance(tool_inputs, str):
                    agent_decision.tool_input = kwargs['input']
            else:
                agent_decision.return_values['output'] = ''
            return agent_decision
        except OutputParserException:
            return AgentFinish({'output': "I'm sorry, the answer of model is invalid, I don't know how to respond to that."}, '')

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool], prefix: str=PREFIX, suffix: str=SUFFIX, human_message_template: str=HUMAN_MESSAGE_TEMPLATE, format_instructions: str=FORMAT_INSTRUCTIONS, input_variables: Optional[List[str]]=None, memory_prompts: Optional[List[BasePromptTemplate]]=None) -> BasePromptTemplate:
        if False:
            i = 10
            return i + 15
        tool_strings = []
        for tool in tools:
            args_schema = re.sub('}', '}}}}', re.sub('{', '{{{{', str(tool.args)))
            tool_strings.append(f'{tool.name}: {tool.description}, args: {args_schema}')
        formatted_tools = '\n'.join(tool_strings)
        unique_tool_names = set((tool.name for tool in tools))
        tool_names = ', '.join(('"' + name + '"' for name in unique_tool_names))
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = '\n\n'.join([prefix, formatted_tools, format_instructions, suffix])
        if input_variables is None:
            input_variables = ['input', 'agent_scratchpad']
        _memory_prompts = memory_prompts or []
        messages = [SystemMessagePromptTemplate.from_template(template), *_memory_prompts, HumanMessagePromptTemplate.from_template(human_message_template)]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    @classmethod
    def create_completion_prompt(cls, tools: Sequence[BaseTool], prefix: str=PREFIX, format_instructions: str=FORMAT_INSTRUCTIONS, input_variables: Optional[List[str]]=None) -> PromptTemplate:
        if False:
            return 10
        'Create prompt in the style of the zero shot agent.\n\n        Args:\n            tools: List of tools the agent will have access to, used to format the\n                prompt.\n            prefix: String to put before the list of tools.\n            input_variables: List of input variables the final prompt will expect.\n\n        Returns:\n            A PromptTemplate with the template assembled from the pieces here.\n        '
        suffix = 'Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.\nQuestion: {input}\nThought: {agent_scratchpad}\n'
        tool_strings = '\n'.join([f'{tool.name}: {tool.description}' for tool in tools])
        tool_names = ', '.join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        template = '\n\n'.join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ['input', 'agent_scratchpad']
        return PromptTemplate(template=template, input_variables=input_variables)

    def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        if False:
            return 10
        agent_scratchpad = ''
        for (action, observation) in intermediate_steps:
            agent_scratchpad += action.log
            agent_scratchpad += f'\n{self.observation_prefix}{observation}\n{self.llm_prefix}'
        if not isinstance(agent_scratchpad, str):
            raise ValueError('agent_scratchpad should be of type string.')
        if agent_scratchpad:
            llm_chain = cast(LLMChain, self.llm_chain)
            if llm_chain.model_instance.model_mode == ModelMode.CHAT:
                return f"This was your previous work (but I haven't seen any of it! I only see what you return as final answer):\n{agent_scratchpad}"
            else:
                return agent_scratchpad
        else:
            return agent_scratchpad

    @classmethod
    def from_llm_and_tools(cls, model_instance: BaseLLM, tools: Sequence[BaseTool], callback_manager: Optional[BaseCallbackManager]=None, output_parser: Optional[AgentOutputParser]=None, prefix: str=PREFIX, suffix: str=SUFFIX, human_message_template: str=HUMAN_MESSAGE_TEMPLATE, format_instructions: str=FORMAT_INSTRUCTIONS, input_variables: Optional[List[str]]=None, memory_prompts: Optional[List[BasePromptTemplate]]=None, **kwargs: Any) -> Agent:
        if False:
            print('Hello World!')
        'Construct an agent from an LLM and tools.'
        cls._validate_tools(tools)
        if model_instance.model_mode == ModelMode.CHAT:
            prompt = cls.create_prompt(tools, prefix=prefix, suffix=suffix, human_message_template=human_message_template, format_instructions=format_instructions, input_variables=input_variables, memory_prompts=memory_prompts)
        else:
            prompt = cls.create_completion_prompt(tools, prefix=prefix, format_instructions=format_instructions, input_variables=input_variables)
        llm_chain = LLMChain(model_instance=model_instance, prompt=prompt, callback_manager=callback_manager)
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, output_parser=_output_parser, dataset_tools=tools, **kwargs)