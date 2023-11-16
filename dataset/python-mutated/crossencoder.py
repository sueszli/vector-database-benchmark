"""
CrossEncoder module
"""
import numpy as np
from ..hfpipeline import HFPipeline

class CrossEncoder(HFPipeline):
    """
    Computes similarity between query and list of text using a cross-encoder model
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__('text-classification', path, quantize, gpu, model, **kwargs)

    def __call__(self, query, texts, multilabel=True, workers=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the similarity between query and list of text. Returns a list of\n        (id, score) sorted by highest score, where id is the index in texts.\n\n        This method supports query as a string or a list. If the input is a string,\n        the return type is a 1D list of (id, score). If text is a list, a 2D list\n        of (id, score) is returned with a row per string.\n\n        Args:\n            query: query text|list\n            texts: list of text\n            multilabel: labels are independent if True, scores are normalized to sum to 1 per text item if False, raw scores returned if None\n            workers: number of concurrent workers to use for processing data, defaults to None\n\n        Returns:\n            list of (id, score)\n        '
        scores = []
        for q in [query] if isinstance(query, str) else query:
            result = self.pipeline([{'text': q, 'text_pair': t} for t in texts], top_k=None, function_to_apply='none', num_workers=workers)
            scores.append(self.function([r[0]['score'] for r in result], multilabel))
        scores = [sorted(enumerate(row), key=lambda x: x[1], reverse=True) for row in scores]
        return scores[0] if isinstance(query, str) else scores

    def function(self, scores, multilabel):
        if False:
            print('Hello World!')
        '\n        Applys an output transformation function based on value of multilabel.\n\n        Args:\n            scores: input scores\n            multilabel: labels are independent if True, scores are normalized to sum to 1 per text item if False, raw scores returned if None\n\n        Returns:\n            transformed scores\n        '
        identity = lambda x: x
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x))
        function = identity if multilabel is None else sigmoid if multilabel else softmax
        return function(np.array(scores))