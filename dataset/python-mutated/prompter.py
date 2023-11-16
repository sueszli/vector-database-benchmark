import json
import os.path as osp
from typing import Union
from bigdl.llm.utils.common import invalidInputError

class Prompter(object):
    __slots__ = ('template', '_verbose')

    def __init__(self, template_name: str='', verbose: bool=False):
        if False:
            while True:
                i = 10
        self._verbose = verbose
        if not template_name:
            template_name = 'alpaca'
        file_name = osp.join('templates', f'{template_name}.json')
        if not osp.exists(file_name):
            invalidInputError(False, f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(self, instruction: str, input: Union[None, str]=None, label: Union[None, str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        if input:
            res = self.template['prompt_input'].format(instruction=instruction, input=input)
        else:
            res = self.template['prompt_no_input'].format(instruction=instruction)
        if label:
            res = f'{res}{label}'
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        if False:
            i = 10
            return i + 15
        return output.split(self.template['response_split'])[1].strip()