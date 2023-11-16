import re
from typing import Type, Optional, List
from pydantic import BaseModel, Field
from superagi.agent.agent_prompt_builder import AgentPromptBuilder
from superagi.helper.error_handler import ErrorHandler
from superagi.helper.prompt_reader import PromptReader
from superagi.helper.token_counter import TokenCounter
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm
from superagi.models.agent_execution import AgentExecution
from superagi.models.agent_execution_feed import AgentExecutionFeed
from superagi.resource_manager.file_manager import FileManager
from superagi.tools.base_tool import BaseTool
from superagi.tools.tool_response_query_manager import ToolResponseQueryManager

class ImproveCodeSchema(BaseModel):
    pass

class ImproveCodeTool(BaseTool):
    """
    Used to improve the already generated code by reading the code from the files

    Attributes:
        llm: LLM used for code generation.
        name : The name of the tool.
        description : The description of the tool.
        resource_manager: Manages the file resources.
    """
    llm: Optional[BaseLlm] = None
    agent_id: int = None
    agent_execution_id: int = None
    name = 'ImproveCodeTool'
    description = 'This tool improves the generated code.'
    args_schema: Type[ImproveCodeSchema] = ImproveCodeSchema
    resource_manager: Optional[FileManager] = None
    tool_response_manager: Optional[ToolResponseQueryManager] = None
    goals: List[str] = []

    class Config:
        arbitrary_types_allowed = True

    def _execute(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Execute the improve code tool.\n\n        Returns:\n            Improved code or error message.\n        '
        file_names = self.resource_manager.get_files()
        logger.info(file_names)
        for file_name in file_names:
            if '.txt' not in file_name and '.sh' not in file_name and ('.json' not in file_name):
                content = self.resource_manager.read_file(file_name)
                prompt = PromptReader.read_tools_prompt(__file__, 'improve_code.txt')
                prompt = prompt.replace('{goals}', AgentPromptBuilder.add_list_items_to_string(self.goals))
                prompt = prompt.replace('{content}', content)
                prompt = prompt + '\nOriginal Code:\n```\n' + content + '\n```'
                result = self.llm.chat_completion([{'role': 'system', 'content': prompt}])
                if result is not None and 'error' in result and (result['message'] is not None):
                    ErrorHandler.handle_openai_errors(self.toolkit_config.session, self.agent_id, self.agent_execution_id, result['message'])
                response = result.get('response')
                if not response:
                    logger.info('RESPONSE NOT AVAILABLE')
                choices = response.get('choices')
                if not choices:
                    logger.info('CHOICES NOT AVAILABLE')
                improved_content = choices[0]['message']['content']
                parsed_content = re.findall('```(?:\\w*\n)?(.*?)```', improved_content, re.DOTALL)
                parsed_content_code = '\n'.join(parsed_content)
                save_result = self.resource_manager.write_file(file_name, parsed_content_code)
                if save_result.startswith('Error'):
                    return save_result
            else:
                continue
        return f'All codes improved and saved successfully in: ' + ' '.join(file_names)