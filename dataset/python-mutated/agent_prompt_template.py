import re
from pydantic.types import List
from superagi.helper.prompt_reader import PromptReader
FINISH_NAME = 'finish'

class AgentPromptTemplate:

    @staticmethod
    def add_list_items_to_string(items: List[str]) -> str:
        if False:
            i = 10
            return i + 15
        list_string = ''
        for (i, item) in enumerate(items):
            list_string += f'{i + 1}. {item}\n'
        return list_string

    @classmethod
    def clean_prompt(cls, prompt):
        if False:
            while True:
                i = 10
        prompt = re.sub('[ \t]+', ' ', prompt)
        return prompt.strip()

    @classmethod
    def get_super_agi_single_prompt(cls):
        if False:
            for i in range(10):
                print('nop')
        super_agi_prompt = PromptReader.read_agent_prompt(__file__, 'superagi.txt')
        return {'prompt': super_agi_prompt, 'variables': ['goals', 'instructions', 'constraints', 'tools']}

    @classmethod
    def start_task_based(cls):
        if False:
            i = 10
            return i + 15
        super_agi_prompt = PromptReader.read_agent_prompt(__file__, 'initialize_tasks.txt')
        return {'prompt': AgentPromptTemplate.clean_prompt(super_agi_prompt), 'variables': ['goals', 'instructions']}

    @classmethod
    def analyse_task(cls):
        if False:
            print('Hello World!')
        constraints = ['Exclusively use the tools listed in double quotes e.g. "tool name"']
        super_agi_prompt = PromptReader.read_agent_prompt(__file__, 'analyse_task.txt')
        super_agi_prompt = AgentPromptTemplate.clean_prompt(super_agi_prompt).replace('{constraints}', AgentPromptTemplate.add_list_items_to_string(constraints))
        return {'prompt': super_agi_prompt, 'variables': ['goals', 'instructions', 'tools', 'current_task']}

    @classmethod
    def create_tasks(cls):
        if False:
            return 10
        super_agi_prompt = PromptReader.read_agent_prompt(__file__, 'create_tasks.txt')
        return {'prompt': AgentPromptTemplate.clean_prompt(super_agi_prompt), 'variables': ['goals', 'instructions', 'last_task', 'last_task_result', 'pending_tasks']}

    @classmethod
    def prioritize_tasks(cls):
        if False:
            while True:
                i = 10
        super_agi_prompt = PromptReader.read_agent_prompt(__file__, 'prioritize_tasks.txt')
        return {'prompt': AgentPromptTemplate.clean_prompt(super_agi_prompt), 'variables': ['goals', 'instructions', 'last_task', 'last_task_result', 'pending_tasks']}