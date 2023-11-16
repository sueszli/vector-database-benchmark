from collections import namedtuple
from pyro.distributions.util import scale_and_mask

class ScoreParts(namedtuple('ScoreParts', ['log_prob', 'score_function', 'entropy_term'])):
    """
    This data structure stores terms used in stochastic gradient estimators that
    combine the pathwise estimator and the score function estimator.
    """

    def scale_and_mask(self, scale=1.0, mask=None):
        if False:
            print('Hello World!')
        '\n        Scale and mask appropriate terms of a gradient estimator by a data multiplicity factor.\n        Note that the `score_function` term should not be scaled or masked.\n\n        :param scale: a positive scale\n        :type scale: torch.Tensor or number\n        :param mask: an optional masking tensor\n        :type mask: torch.BoolTensor or None\n        '
        log_prob = scale_and_mask(self.log_prob, scale, mask)
        score_function = self.score_function
        entropy_term = scale_and_mask(self.entropy_term, scale, mask)
        return ScoreParts(log_prob, score_function, entropy_term)