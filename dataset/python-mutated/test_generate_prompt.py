import unittest
from string import Template
from embedchain import App
from embedchain.config import AppConfig, BaseLlmConfig

class TestGeneratePrompt(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.app = App(config=AppConfig(collect_metrics=False))

    def test_generate_prompt_with_template(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that the generate_prompt method correctly formats the prompt using\n        a custom template provided in the BaseLlmConfig instance.\n\n        This test sets up a scenario with an input query and a list of contexts,\n        and a custom template, and then calls generate_prompt. It checks that the\n        returned prompt correctly incorporates all the contexts and the query into\n        the format specified by the template.\n        '
        input_query = 'Test query'
        contexts = ['Context 1', 'Context 2', 'Context 3']
        template = 'You are a bot. Context: ${context} - Query: ${query} - Helpful answer:'
        config = BaseLlmConfig(template=Template(template))
        self.app.llm.config = config
        result = self.app.llm.generate_prompt(input_query, contexts)
        expected_result = 'You are a bot. Context: Context 1 | Context 2 | Context 3 - Query: Test query - Helpful answer:'
        self.assertEqual(result, expected_result)

    def test_generate_prompt_with_contexts_list(self):
        if False:
            print('Hello World!')
        '\n        Tests that the generate_prompt method correctly handles a list of contexts.\n\n        This test sets up a scenario with an input query and a list of contexts,\n        and then calls generate_prompt. It checks that the returned prompt\n        correctly includes all the contexts and the query.\n        '
        input_query = 'Test query'
        contexts = ['Context 1', 'Context 2', 'Context 3']
        config = BaseLlmConfig()
        self.app.llm.config = config
        result = self.app.llm.generate_prompt(input_query, contexts)
        expected_result = config.template.substitute(context='Context 1 | Context 2 | Context 3', query=input_query)
        self.assertEqual(result, expected_result)

    def test_generate_prompt_with_history(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test the 'generate_prompt' method with BaseLlmConfig containing a history attribute.\n        "
        config = BaseLlmConfig()
        config.template = Template('Context: $context | Query: $query | History: $history')
        self.app.llm.config = config
        self.app.llm.set_history(['Past context 1', 'Past context 2'])
        prompt = self.app.llm.generate_prompt('Test query', ['Test context'])
        expected_prompt = "Context: Test context | Query: Test query | History: ['Past context 1', 'Past context 2']"
        self.assertEqual(prompt, expected_prompt)