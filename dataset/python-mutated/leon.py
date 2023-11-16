import random
import sys
from typing import Union
from time import sleep
import json
from .types import AnswerInput, AnswerData, AnswerConfig
from ..constants import SKILL_CONFIG, INTENT_OBJECT

class Leon:
    instance: 'Leon' = None

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        if not Leon.instance:
            Leon.instance = self

    @staticmethod
    def set_answer_data(answer_key: str, data: Union[AnswerData, None]=None) -> Union[str, AnswerConfig]:
        if False:
            while True:
                i = 10
        '\n        Apply data to the answer\n        :param answer_key: The answer key\n        :param data: The data to apply\n        '
        try:
            if SKILL_CONFIG.get('answers') is None or SKILL_CONFIG['answers'].get(answer_key) is None:
                return answer_key
            answers = SKILL_CONFIG['answers'].get(answer_key, '')
            if isinstance(answers, list):
                answer = answers[random.randrange(len(answers))]
            else:
                answer = answers
            if data:
                for (key, value) in data.items():
                    if not isinstance(answer, str) and answer.get('text'):
                        answer['text'] = answer['text'].replace('%{}%'.format(key), str(value))
                        answer['speech'] = answer['speech'].replace('%{}%'.format(key), str(value))
                    else:
                        answer = answer.replace('%{}%'.format(key), str(value))
            if SKILL_CONFIG.get('variables'):
                for (key, value) in SKILL_CONFIG['variables'].items():
                    if not isinstance(answer, str) and answer.get('text'):
                        answer['text'] = answer['text'].replace('%{}%'.format(key), str(value))
                        answer['speech'] = answer['speech'].replace('%{}%'.format(key), str(value))
                    else:
                        answer = answer.replace('%{}%'.format(key), str(value))
            return answer
        except Exception as e:
            print('Error while setting answer data:', e)
            raise e

    def answer(self, answer_input: AnswerInput) -> None:
        if False:
            print('Hello World!')
        '\n        Send an answer to the core\n        :param answer_input: The answer input\n        '
        try:
            key = answer_input.get('key')
            output = {'output': {'codes': 'widget' if answer_input.get('widget') and (not answer_input.get('key')) else answer_input.get('key'), 'answer': self.set_answer_data(key, answer_input.get('data')) if key is not None else '', 'core': answer_input.get('core')}}
            if answer_input.get('widget'):
                output['output']['widget'] = answer_input['widget'].__dict__
            answer_object = {**INTENT_OBJECT, **output}
            sleep(0.1)
            sys.stdout.write(json.dumps(answer_object))
            sys.stdout.flush()
        except Exception as e:
            print('Error while creating answer:', e)
leon = Leon()