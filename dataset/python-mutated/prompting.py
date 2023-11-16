"""
Relative to this file I will have a prompt directory its located ../prompts
In this directory there will be a techniques directory and a directory for each model - gpt-3.5-turbo gpt-4, llama-2-70B, code-llama-7B etc

Each directory will have jinga2 templates for the prompts.
prompts in the model directories can use the techniques in the techniques directory.

Write the code I'd need to load and populate the templates.

I want the following functions:

class PromptEngine:

    def __init__(self, model):
        pass

    def load_prompt(model, prompt_name, prompt_ags) -> str:
        pass
"""
import glob
import os
from difflib import get_close_matches
from typing import List
from jinja2 import Environment, FileSystemLoader
from .forge_log import ForgeLogger
LOG = ForgeLogger(__name__)

class PromptEngine:
    """
    Class to handle loading and populating Jinja2 templates for prompts.
    """

    def __init__(self, model: str, debug_enabled: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the PromptEngine with the specified model.\n\n        Args:\n            model (str): The model to use for loading prompts.\n            debug_enabled (bool): Enable or disable debug logging.\n        '
        self.model = model
        self.debug_enabled = debug_enabled
        if self.debug_enabled:
            LOG.debug(f'Initializing PromptEngine for model: {model}')
        try:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../prompts'))
            model_names = [os.path.basename(os.path.normpath(d)) for d in glob.glob(os.path.join(models_dir, '*/')) if os.path.isdir(d) and 'techniques' not in d]
            self.model = self.get_closest_match(self.model, model_names)
            if self.debug_enabled:
                LOG.debug(f'Using the closest match model for prompts: {self.model}')
            self.env = Environment(loader=FileSystemLoader(models_dir))
        except Exception as e:
            LOG.error(f'Error initializing Environment: {e}')
            raise

    @staticmethod
    def get_closest_match(target: str, model_dirs: List[str]) -> str:
        if False:
            return 10
        '\n        Find the closest match to the target in the list of model directories.\n\n        Args:\n            target (str): The target model.\n            model_dirs (list): The list of available model directories.\n\n        Returns:\n            str: The closest match to the target.\n        '
        try:
            matches = get_close_matches(target, model_dirs, n=1, cutoff=0.1)
            if matches:
                matches_str = ', '.join(matches)
                LOG.debug(matches_str)
            for m in matches:
                LOG.info(m)
            return matches[0]
        except Exception as e:
            LOG.error(f'Error finding closest match: {e}')
            raise

    def load_prompt(self, template: str, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Load and populate the specified template.\n\n        Args:\n            template (str): The name of the template to load.\n            **kwargs: The arguments to populate the template with.\n\n        Returns:\n            str: The populated template.\n        '
        try:
            template = os.path.join(self.model, template)
            if self.debug_enabled:
                LOG.debug(f'Loading template: {template}')
            template = self.env.get_template(f'{template}.j2')
            if self.debug_enabled:
                LOG.debug(f'Rendering template: {template} with args: {kwargs}')
            return template.render(**kwargs)
        except Exception as e:
            LOG.error(f'Error loading or rendering template: {e}')
            raise