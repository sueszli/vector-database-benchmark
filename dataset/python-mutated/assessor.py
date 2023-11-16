"""
Assessor analyzes trial's intermediate results (e.g., periodically evaluated accuracy on test dataset)
to tell whether this trial can be early stopped or not.

See :class:`Assessor`' specification and ``docs/en_US/assessors.rst`` for details.
"""
from __future__ import annotations
from enum import Enum
import logging
from .recoverable import Recoverable
from .typehint import TrialMetric
__all__ = ['AssessResult', 'Assessor']
_logger = logging.getLogger(__name__)

class AssessResult(Enum):
    """
    Enum class for :meth:`Assessor.assess_trial` return value.
    """
    Good = True
    'The trial works well.'
    Bad = False
    'The trial works poorly and should be early stopped.'

class Assessor(Recoverable):
    """
    Assessor analyzes trial's intermediate results (e.g., periodically evaluated accuracy on test dataset)
    to tell whether this trial can be early stopped or not.

    This is the abstract base class for all assessors.
    Early stopping algorithms should inherit this class and override :meth:`assess_trial` method,
    which receives intermediate results from trials and give an assessing result.

    If :meth:`assess_trial` returns :obj:`AssessResult.Bad` for a trial,
    it hints NNI framework that the trial is likely to result in a poor final accuracy,
    and therefore should be killed to save resource.

    If an assessor want's to be notified when a trial ends, it can also override :meth:`trial_end`.

    To write a new assessor, you can reference :class:`~nni.medianstop_assessor.MedianstopAssessor`'s code as an example.

    See Also
    --------
    Builtin assessors:
    :class:`~nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor`
    :class:`~nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor`
    """

    def assess_trial(self, trial_job_id: str, trial_history: list[TrialMetric]) -> AssessResult:
        if False:
            for i in range(10):
                print('nop')
        "\n        Abstract method for determining whether a trial should be killed. Must override.\n\n        The NNI framework has little guarantee on ``trial_history``.\n        This method is not guaranteed to be invoked for each time ``trial_history`` get updated.\n        It is also possible that a trial's history keeps updating after receiving a bad result.\n        And if the trial failed and retried, ``trial_history`` may be inconsistent with its previous value.\n\n        The only guarantee is that ``trial_history`` is always growing.\n        It will not be empty and will always be longer than previous value.\n\n        This is an example of how :meth:`assess_trial` get invoked sequentially:\n\n        ::\n\n            trial_job_id | trial_history   | return value\n            ------------ | --------------- | ------------\n            Trial_A      | [1.0, 2.0]      | Good\n            Trial_B      | [1.5, 1.3]      | Bad\n            Trial_B      | [1.5, 1.3, 1.9] | Good\n            Trial_A      | [0.9, 1.8, 2.3] | Good\n\n        Parameters\n        ----------\n        trial_job_id : str\n            Unique identifier of the trial.\n        trial_history : list\n            Intermediate results of this trial. The element type is decided by trial code.\n\n        Returns\n        -------\n        AssessResult\n            :obj:`AssessResult.Good` or :obj:`AssessResult.Bad`.\n        "
        raise NotImplementedError('Assessor: assess_trial not implemented')

    def trial_end(self, trial_job_id: str, success: bool) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Abstract method invoked when a trial is completed or terminated. Do nothing by default.\n\n        Parameters\n        ----------\n        trial_job_id : str\n            Unique identifier of the trial.\n        success : bool\n            True if the trial successfully completed; False if failed or terminated.\n        '

    def load_checkpoint(self) -> None:
        if False:
            return 10
        '\n        Internal API under revising, not recommended for end users.\n        '
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Load checkpoint ignored by assessor, checkpoint path: %s', checkpoin_path)

    def save_checkpoint(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Internal API under revising, not recommended for end users.\n        '
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Save checkpoint ignored by assessor, checkpoint path: %s', checkpoin_path)

    def _on_exit(self) -> None:
        if False:
            return 10
        pass

    def _on_error(self) -> None:
        if False:
            while True:
                i = 10
        pass