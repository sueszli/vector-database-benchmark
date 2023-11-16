"""
Questions module
"""
from ..hfpipeline import HFPipeline

class Questions(HFPipeline):
    """
    Runs extractive QA for a series of questions and contexts.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__('question-answering', path, quantize, gpu, model, **kwargs)

    def __call__(self, questions, contexts, workers=0):
        if False:
            print('Hello World!')
        '\n        Runs a extractive question-answering model against each question-context pair, finding the best answers.\n\n        Args:\n            questions: list of questions\n            contexts: list of contexts to pull answers from\n            workers: number of concurrent workers to use for processing data, defaults to None\n\n        Returns:\n            list of answers\n        '
        answers = []
        for (x, question) in enumerate(questions):
            if question and contexts[x]:
                result = self.pipeline(question=question, context=contexts[x], num_workers=workers)
                (answer, score) = (result['answer'], result['score'])
                if score < 0.05:
                    answer = None
                answers.append(answer)
            else:
                answers.append(None)
        return answers