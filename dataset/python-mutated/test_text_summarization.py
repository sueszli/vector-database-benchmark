import unittest
from transformers import load_tool
from .test_tools_common import ToolTesterMixin
TEXT = '\nHugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf originally as a company that developed a chatbot app targeted at teenagers.[2] After open-sourcing the model behind the chatbot, the company pivoted to focus on being a platform for machine learning.\n\nIn March 2021, Hugging Face raised $40 million in a Series B funding round.[3]\n\nOn April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model.[4] In 2022, the workshop concluded with the announcement of BLOOM, a multilingual large language model with 176 billion parameters.[5]\n'

class TextSummarizationToolTester(unittest.TestCase, ToolTesterMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tool = load_tool('summarization')
        self.tool.setup()
        self.remote_tool = load_tool('summarization', remote=True)

    def test_exact_match_arg(self):
        if False:
            print('Hello World!')
        result = self.tool(TEXT)
        self.assertEqual(result, 'Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf. In March 2021, Hugging Face raised $40 million in a Series B funding round. On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model. In 2022, the workshop concluded with the announcement of BLOOM.')

    def test_exact_match_arg_remote(self):
        if False:
            print('Hello World!')
        result = self.remote_tool(TEXT)
        self.assertEqual(result, 'Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf. In March 2021, Hugging Face raised $40 million in a Series B funding round. On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model. In 2022, the workshop concluded with the announcement of BLOOM.')

    def test_exact_match_kwarg(self):
        if False:
            i = 10
            return i + 15
        result = self.tool(text=TEXT)
        self.assertEqual(result, 'Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf. In March 2021, Hugging Face raised $40 million in a Series B funding round. On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model. In 2022, the workshop concluded with the announcement of BLOOM.')

    def test_exact_match_kwarg_remote(self):
        if False:
            i = 10
            return i + 15
        result = self.remote_tool(text=TEXT)
        self.assertEqual(result, 'Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf. In March 2021, Hugging Face raised $40 million in a Series B funding round. On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model. In 2022, the workshop concluded with the announcement of BLOOM.')