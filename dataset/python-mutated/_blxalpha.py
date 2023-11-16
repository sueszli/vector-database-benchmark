import numpy as np
from optuna._experimental import experimental_class
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.study import Study

@experimental_class('3.0.0')
class BLXAlphaCrossover(BaseCrossover):
    """Blend Crossover operation used by :class:`~optuna.samplers.NSGAIISampler`.

    Uniformly samples child individuals from the hyper-rectangles created
    by the two parent individuals. For further information about BLX-alpha crossover,
    please refer to the following paper:

    - `Eshelman, L. and J. D. Schaffer.
      Real-Coded Genetic Algorithms and Interval-Schemata. FOGA (1992).
      <https://www.sciencedirect.com/science/article/abs/pii/B9780080948324500180>`_

    Args:
        alpha:
            Parametrizes blend operation.
    """
    n_parents = 2

    def __init__(self, alpha: float=0.5) -> None:
        if False:
            while True:
                i = 10
        self._alpha = alpha

    def crossover(self, parents_params: np.ndarray, rng: np.random.RandomState, study: Study, search_space_bounds: np.ndarray) -> np.ndarray:
        if False:
            return 10
        parents_min = parents_params.min(axis=0)
        parents_max = parents_params.max(axis=0)
        diff = self._alpha * (parents_max - parents_min)
        low = parents_min - diff
        high = parents_max + diff
        r = rng.rand(len(search_space_bounds))
        child_params = (high - low) * r + low
        return child_params