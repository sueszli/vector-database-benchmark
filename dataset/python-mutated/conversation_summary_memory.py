from typing import Optional, Union, Dict, Any, List
from haystack.agents.memory import ConversationMemory
from haystack.nodes import PromptTemplate, PromptNode

class ConversationSummaryMemory(ConversationMemory):
    """
    A memory class that stores conversation history and periodically generates summaries.
    """

    def __init__(self, prompt_node: PromptNode, prompt_template: Optional[Union[str, PromptTemplate]]=None, input_key: str='input', output_key: str='output', summary_frequency: int=3):
        if False:
            return 10
        '\n        Initialize ConversationSummaryMemory with a PromptNode, optional prompt_template,\n        input and output keys, and a summary_frequency.\n\n        :param prompt_node: A PromptNode object for generating conversation summaries.\n        :param prompt_template: Optional prompt template as a string or PromptTemplate object.\n        :param input_key: input key, default is "input".\n        :param output_key: output key, default is "output".\n        :param summary_frequency: integer specifying how often to generate a summary (default is 3).\n        '
        super().__init__(input_key, output_key)
        self.save_count = 0
        self.prompt_node = prompt_node
        template = prompt_template if prompt_template is not None else prompt_node.default_prompt_template or 'conversational-summary'
        self.template = prompt_node.get_prompt_template(template)
        self.summary_frequency = summary_frequency
        self.summary = ''

    def load(self, keys: Optional[List[str]]=None, **kwargs) -> str:
        if False:
            print('Hello World!')
        '\n        Load conversation history as a formatted string, including the latest summary.\n\n        :param keys: Optional list of keys (ignored in this implementation).\n        :param kwargs: Optional keyword arguments\n            - window_size: integer specifying the number of most recent conversation snippets to load.\n        :return: A formatted string containing the conversation history with the latest summary.\n        '
        if self.has_unsummarized_snippets():
            unsummarized = self.load_recent_snippets(window_size=self.unsummarized_snippets())
            return f'{self.summary}\n{unsummarized}'
        else:
            return self.summary

    def load_recent_snippets(self, window_size: int=1) -> str:
        if False:
            print('Hello World!')
        '\n        Load the most recent conversation snippets as a formatted string.\n\n        :param window_size: integer specifying the number of most recent conversation snippets to load.\n        :return: A formatted string containing the most recent conversation snippets.\n        '
        return super().load(window_size=window_size)

    def summarize(self) -> str:
        if False:
            return 10
        '\n        Generate a summary of the conversation history and clear the history.\n\n        :return: A string containing the generated summary.\n        '
        most_recent_chat_snippets = self.load_recent_snippets(window_size=self.summary_frequency)
        pn_response = self.prompt_node.prompt(self.template, chat_transcript=most_recent_chat_snippets)
        return pn_response[0]

    def needs_summary(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Determine if a new summary should be generated.\n\n        :return: True if a new summary should be generated, otherwise False.\n        '
        return self.save_count % self.summary_frequency == 0

    def unsummarized_snippets(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns how many conversation snippets have not been summarized.\n        :return: The number of conversation snippets that have not been summarized.\n        '
        return self.save_count % self.summary_frequency

    def has_unsummarized_snippets(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if there are any conversation snippets that have not been summarized.\n        :return: True if there are unsummarized snippets, otherwise False.\n        '
        return self.unsummarized_snippets() != 0

    def save(self, data: Dict[str, Any]) -> None:
        if False:
            return 10
        '\n        Save a conversation snippet to memory and update the save count.\n        Generate a summary if needed.\n\n        :param data: A dictionary containing the conversation snippet to save.\n        '
        super().save(data)
        self.save_count += 1
        if self.needs_summary():
            self.summary += self.summarize()

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Clear the conversation history and the summary.\n        '
        super().clear()
        self.save_count = 0
        self.summary = ''