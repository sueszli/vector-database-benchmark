""" This module parses a json/yaml file that defines a flow of questions to fulfill the cookiecutter context"""
from typing import Dict, Optional, Tuple
import yaml
from samcli.commands.exceptions import UserException
from samcli.yamlhelper import parse_yaml_file
from .interactive_flow import InteractiveFlow
from .question import Question, QuestionFactory

class QuestionsNotFoundException(UserException):
    pass

class QuestionsFailedParsingException(UserException):
    pass

class InteractiveFlowCreator:

    @staticmethod
    def create_flow(flow_definition_path: str, extra_context: Optional[Dict]=None) -> InteractiveFlow:
        if False:
            while True:
                i = 10
        '\n        This method parses the given json/yaml file to create an InteractiveFLow. It expects the file to define\n        a list of questions. It parses the questions and add it to the flow in the same order they are defined\n        in the file, i.e. the default next-question of a given question will be the next one defined in the file,\n        while also respecting the question-defined _next_question map if provided.\n\n        Parameters:\n        ----------\n        flow_definition_path: str\n            A path to a json/yaml file that defines the questions of the flow. the file is expected to be in the\n            following format:\n            {\n                "questions": [\n                    {\n                      "key": "key of the corresponding cookiecutter config",\n                      "question": "the question to prompt to the user",\n                      "kind": "the kind of the question, for example: confirm",\n                      "isRequired": true/false,\n                      # optional branching logic to jump to a particular question based on the answer. if not given\n                      # will automatically go to next question\n                      "nextQuestion": {\n                        "True": "key of the question to jump to if the user answered \'Yes\'",\n                        "False": "key of the question to jump to if the user answered \'Yes\'",\n                      }\n                      "default": "default_answer",\n                      # the default value can also be loaded from cookiecutter context\n                      # with a key path whose key path item can be loaded from cookiecutter as well.\n                      "default": {\n                        "keyPath": [\n                            {\n                                "valueOf": "key-of-another-question"\n                            },\n                            "pipeline_user"\n                        ]\n                      }\n                      # assuming the answer of "key-of-another-question" is "ABC"\n                      # the default value will be load from cookiecutter context with key "[\'ABC\', \'pipeline_user]"\n                    },\n                    ...\n                ]\n            }\n        extra_context: Dict\n            if the template contains variable($variableName) this parameter provides the values for those variables.\n\n        Returns: InteractiveFlow(questions={k1: q1, k2: q2, ...}, first_question_key="first question\'s key")\n        '
        (questions, first_question_key) = InteractiveFlowCreator._load_questions(flow_definition_path, extra_context)
        return InteractiveFlow(questions=questions, first_question_key=first_question_key)

    @staticmethod
    def _load_questions(flow_definition_path: str, extra_context: Optional[Dict]=None) -> Tuple[Dict[str, Question], str]:
        if False:
            return 10
        previous_question: Optional[Question] = None
        first_question_key: str = ''
        questions: Dict[str, Question] = {}
        questions_definition = InteractiveFlowCreator._parse_questions_definition(flow_definition_path, extra_context)
        try:
            for question in questions_definition.get('questions', []):
                q = QuestionFactory.create_question_from_json(question)
                if not first_question_key:
                    first_question_key = q.key
                elif previous_question and (not previous_question.default_next_question_key):
                    previous_question.set_default_next_question_key(q.key)
                questions[q.key] = q
                previous_question = q
            return (questions, first_question_key)
        except (KeyError, ValueError, AttributeError, TypeError) as ex:
            raise QuestionsFailedParsingException(f'Failed to parse questions: {str(ex)}') from ex

    @staticmethod
    def _parse_questions_definition(file_path: str, extra_context: Optional[Dict]=None) -> Dict:
        if False:
            return 10
        "\n        Read the questions definition file, do variable substitution, parse it as JSON/YAML\n\n        Parameters\n        ----------\n        file_path : string\n            Path to the questions definition to read\n        extra_context : Dict\n            if the file contains variable($variableName) this parameter provides the values for those variables.\n\n        Raises\n        ------\n        QuestionsNotFoundException: if the file_path doesn't exist\n        QuestionsFailedParsingException: if any error occurred during variables substitution or content parsing\n\n        Returns\n        -------\n        questions data as a dictionary\n        "
        try:
            return parse_yaml_file(file_path=file_path, extra_context=extra_context)
        except FileNotFoundError as ex:
            raise QuestionsNotFoundException(f'questions definition file not found at {file_path}') from ex
        except (KeyError, ValueError, yaml.YAMLError) as ex:
            raise QuestionsFailedParsingException(f'Failed to parse questions: {str(ex)}') from ex