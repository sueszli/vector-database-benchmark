import math
from allennlp.training.metrics.average import Average
from allennlp.training.metrics.metric import Metric

@Metric.register('perplexity')
class Perplexity(Average):
    """
    Perplexity is a common metric used for evaluating how well a language model
    predicts a sample.

    Notes
    -----
    Assumes negative log likelihood loss of each batch (base e). Provides the
    average perplexity of the batches.
    """

    def get_metric(self, reset: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        # Returns\n\n        The accumulated perplexity.\n        '
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            return 0.0
        return math.exp(average_loss)