import asyncio
import logging
import pytest
from azure.ai.generative.synthetic.qa import QADataGenerator, QAType
from devtools_testutils import AzureRecordedTestCase
from utils import find_closest_prediction, f1_score
logger = logging.getLogger(__name__)

@pytest.mark.e2etest
@pytest.mark.live_test_only
@pytest.mark.usefixtures('recorded_test')
class TestQADataGeneratorE2E(AzureRecordedTestCase):

    def validate_token_usage(self, token_usage):
        if False:
            return 10
        assert isinstance(token_usage, dict)
        assert token_usage['completion_tokens'] > 0
        assert token_usage['prompt_tokens'] > 0
        assert token_usage['total_tokens'] > 0

    def test_generation_quality_for_short_answer(self, qa_generator: QADataGenerator):
        if False:
            for i in range(10):
                print('nop')
        text = '\nThe history of the steam engine stretches back as far as the first century AD; the first recorded rudimentary steam engine being the aeolipile described by Hero of Alexandria.\nA rudimentary steam turbine device was described by Taqi al-Din in 1551.\nFast forward a couple of centuries, the US government paid $53,619 for a Mikado steam engine.\n'
        expected_question_answers = {'In what century did the history of steam engines begin?': 'first century AD', 'What was the first recorded rudimentary steam engine as described by Hero of Alexandria?': 'aeolipile', 'Who described the aeolipile, the first recorded rudimentary steam engine?': 'Hero of Alexandria', 'When did Taqi al-Din describe a rudimentary steam turbine device?': '1551', 'Who described a rudimentary steam turbine device in 1551?': 'Taqi al-Din', 'How much did the US government pay for a light Mikado steam engine?': '$53,619', 'What steam engine did the US government purchase for $53,619?': 'Mikado steam engine'}
        num_questions = len(expected_question_answers)
        result = qa_generator.generate(text, QAType.SHORT_ANSWER, num_questions)
        assert len(result['question_answers']) == num_questions
        answer_set = set([a for (_, a) in result['question_answers']])
        expected_answer_set = set(expected_question_answers.values())
        assert answer_set == expected_answer_set, f'Unexpected answer set: {answer_set}'
        expected_questions = list(expected_question_answers.keys())
        for (question, answer) in result['question_answers']:
            closest_question = find_closest_prediction(expected_questions, question)
            expected_answer = expected_question_answers[closest_question]
            assert answer == expected_answer, f'Unexpected answer for question: {closest_question}'
        self.validate_token_usage(result['token_usage'])

    def test_generation_quality_for_long_answer(self, qa_generator: QADataGenerator):
        if False:
            for i in range(10):
                print('nop')
        text = '\nC is a general-purpose computer programming language. It was created by Dennis Ritchie.\nDennis Ritchie also developed the Unix operating system.\nIt was sometimes chosen over interpreted languages for web development because of its speed and near-universal availability.\nIn web development, it used CGI as an information gateway between web application, server, and browser.\n\nFunction parameters are passed by value, although arrays are passed as pointers, i.e. the address of the first item in the array.\n'
        expected_question_answers = [('What is the C programming language and who created it?', 'C is a general-purpose computer programming language created by Dennis Ritchie, who also developed the Unix operating system.'), ('Why was C sometimes chosen over interpreted languages for web development?', 'C was sometimes chosen over interpreted languages for web development because of its speed and near-universal availability. It used CGI as an information gateway between web application, server, and browser.'), ('How are function parameters and arrays passed in the C programming language?', 'In the C programming language, function parameters are passed by value, while arrays are passed as pointers, i.e. the address of the first item in the array.')]
        num_questions = len(expected_question_answers)
        result = qa_generator.generate(text, QAType.LONG_ANSWER, num_questions)
        assert len(result['question_answers']) == num_questions
        f1_min = 0.8
        for (i, (question, answer)) in enumerate(result['question_answers']):
            (expected_question, expected_answer) = expected_question_answers[i]
            f1 = f1_score(question, expected_question)
            assert f1 >= f1_min, f'Question differs from expected. F1:{f1}<{f1_min}\nActual: {question}\nExpected: {expected_question}'
            f1 = f1_score(answer, expected_answer)
            assert f1 >= f1_min, f'Answer differs from expected. F1:{f1}<{f1_min}\nActual: {answer}\nExpected: {expected_answer}'

    def test_generation_quality_for_summary(self, qa_generator: QADataGenerator):
        if False:
            i = 10
            return i + 15
        text = '\nThe history of the steam engine stretches back as far as the first century AD; the first recorded rudimentary steam engine being the aeolipile described by Greek mathematician Hero of Alexandria.\nIn the following centuries, the few steam-powered "engines" known were, like the aeolipile, essentially experimental devices used by inventors to demonstrate the properties of steam.\nA rudimentary steam turbine device was described by Taqi al-Din in 1551 and by Giovanni Branca in 1629. Jerónimo de Ayanz y Beaumont received patents in 1606 for fifty steam powered inventions, including a water pump for draining inundated mines.\n\nNear the end of the 19th century compound engines came into widespread use. Compound engines exhausted steam in to successively larger cylinders to accommodate the higher volumes at reduced pressures, giving improved efficiency.\nThese stages were called expansions, with double and triple expansion engines being common, especially in shipping where efficiency was important to reduce the weight of coal carried. \n\nThe most useful instrument for analyzing the performance of steam engines is the steam engine indicator.\nEarly versions were in use by 1851, but the most successful indicator was developed for the high speed engine inventor and manufacturer Charles Porter by Charles Richard and exhibited at London Exhibition in 1862.\n'
        expected_question_answers = [('Write a summary in 100 words for: History of the Steam Engine', 'The history of steam engines dates back to the first century AD, with the aeolipile described by Greek mathematician Hero of Alexandria.\nOver the centuries, steam-powered engines were primarily experimental devices.\nTaqi al-Din and Giovanni Branca described rudimentary steam turbines in 1551 and 1629, while Jerónimo de Ayanz y Beaumont patented fifty steam-powered inventions in 1606.\nIn the late 19th century, compound engines with multiple expansions improved efficiency, becoming widespread, especially in shipping.\nThe steam engine indicator, developed by Charles Richard for Charles Porter, was a crucial instrument for analyzing steam engine performance.\n')]
        num_questions = len(expected_question_answers)
        result = qa_generator.generate(text, QAType.SUMMARY)
        assert len(result['question_answers']) == num_questions
        (question, answer) = result['question_answers'][0]
        (expected_question, expected_answer) = expected_question_answers[0]
        assert question == expected_question, f'Unexpected question: {expected_question}'
        f1 = f1_score(answer, expected_answer)
        f1_min = 0.8
        assert f1 >= f1_min, f'Answer differs from expected. F1:{f1}<{f1_min}\nActual: {answer}\nExpected: {expected_answer}'
        num_words_min = 120
        assert len(answer.split()) <= num_words_min, f'Summary more than {num_words_min} words: {answer}'

    def test_generation_quality_for_boolean(self, qa_generator: QADataGenerator):
        if False:
            for i in range(10):
                print('nop')
        text = '\nThe history of the steam engine stretches back as far as the first century AD; the first recorded rudimentary steam engine being the aeolipile described by Hero of Alexandria.\nFast forward a couple of centuries, the US government paid $53,619 for a Mikado steam engine.\n\nSteam engine should be started when red button has been enabled.\n'
        expected_question_answers = [('True or false - The history of the steam engine dates back to the first century AD?', 'True'), ('True or false - The first recorded rudimentary steam engine was the aeolipile described by Hero of Alexandria?', 'True'), ('True or false - The US government paid $53,619 for a Mikado steam engine?', 'True'), ('True or false - Steam engine should be started when the red button is disabled?', 'False')]
        num_questions = len(expected_question_answers)
        result = qa_generator.generate(text, QAType.BOOLEAN, num_questions)
        assert len(result['question_answers']) == num_questions
        f1_min = 0.7
        for (i, (question, answer)) in enumerate(result['question_answers']):
            (expected_question, expected_answer) = expected_question_answers[i]
            f1 = f1_score(question, expected_question)
            assert f1 >= f1_min, f'Question differs from expected. F1:{f1}<{f1_min}\nActual: {question}\nExpected: {expected_question}'
            assert answer == expected_answer, 'Answer differs from expected'

    def test_generation_quality_for_conversation(self, qa_generator: QADataGenerator):
        if False:
            while True:
                i = 10
        text = '\nA pipeline endpoint is a collection of published pipelines. This logical organization lets you manage and call multiple pipelines using the same endpoint.\nEach published pipeline in a pipeline endpoint is versioned.\n\nThe Azure Machine Learning CLI is an extension to the Azure CLI, a cross-platform command-line interface for the Azure platform. It has two versions: v1 & v2.\n'
        expected_question_answers = [('What is a pipeline endpoint?', 'A pipeline endpoint is a collection of published pipelines.'), ('What is its purpose?', 'This logical organization lets you manage and call multiple pipelines using the same endpoint.'), ('Are the published pipelines in it versioned?', 'Yes, each published pipeline in a pipeline endpoint is versioned.'), ('What is the Azure Machine Learning CLI?', 'The Azure Machine Learning CLI is an extension to the Azure CLI, a cross-platform command-line interface for the Azure platform.'), ('How many versions does it have?', 'The Azure Machine Learning CLI has two versions: v1 & v2.')]
        num_questions = len(expected_question_answers)
        result = qa_generator.generate(text, QAType.CONVERSATION, num_questions)
        assert len(result['question_answers']) == num_questions
        f1_min = 0.9
        for (i, (question, answer)) in enumerate(result['question_answers']):
            (expected_question, expected_answer) = expected_question_answers[i]
            f1 = f1_score(question, expected_question)
            assert f1 >= f1_min, f'Question differs from expected. F1:{f1}<{f1_min}\nActual: {question}\nExpected: {expected_question}'
            f1 = f1_score(answer, expected_answer)
            assert f1 >= f1_min, f'Answer differs from expected. F1:{f1}<{f1_min}\nActual: {answer}\nExpected: {expected_answer}'
        self.validate_token_usage(result['token_usage'])

    @pytest.mark.asyncio
    async def test_async(self, qa_generator: QADataGenerator):
        text = 'The history of the steam engine stretches back as far as the first century AD.'
        funcs = []
        for _ in range(3):
            funcs.append(qa_generator.generate_async(text, QAType.SHORT_ANSWER, num_questions=1))
        for result in await asyncio.gather(*funcs):
            assert len(result['question_answers']) == 1
            self.validate_token_usage(result['token_usage'])