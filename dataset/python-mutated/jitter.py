import random

class Jitter:
    """Jitter interface"""

    def recalculate(self, duration: float) -> float:
        if False:
            i = 10
            return i + 15
        'Recalculate the given duration.\n        see also: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/\n\n        Args:\n            duration: the duration in seconds\n\n        Returns:\n            A new duration that the jitter amount is added\n        '
        raise NotImplementedError()

class RandomJitter(Jitter):
    """Random jitter implementation"""

    def recalculate(self, duration: float) -> float:
        if False:
            print('Hello World!')
        return duration + random.random()