"""
Tuner is an AutoML algorithm, which generates a new configuration for the next try.
A new trial will run with this configuration.

See :class:`Tuner`' specification and ``docs/en_US/tuners.rst`` for details.
"""
from __future__ import annotations
import logging
import nni
from .recoverable import Recoverable
from .typehint import Parameters, SearchSpace, TrialMetric, TrialRecord
__all__ = ['Tuner']
_logger = logging.getLogger(__name__)

class Tuner(Recoverable):
    """
    Tuner is an AutoML algorithm, which generates a new configuration for the next try.
    A new trial will run with this configuration.

    This is the abstract base class for all tuners.
    Tuning algorithms should inherit this class and override :meth:`update_search_space`, :meth:`receive_trial_result`,
    as well as :meth:`generate_parameters` or :meth:`generate_multiple_parameters`.

    After initializing, NNI will first call :meth:`update_search_space` to tell tuner the feasible region,
    and then call :meth:`generate_parameters` one or more times to request for hyper-parameter configurations.

    The framework will train several models with given configuration.
    When one of them is finished, the final accuracy will be reported to :meth:`receive_trial_result`.
    And then another configuration will be reqeusted and trained, util the whole experiment finish.

    If a tuner want's to know when a trial ends, it can also override :meth:`trial_end`.

    Tuners use *parameter ID* to track trials.
    In tuner context, there is a one-to-one mapping between parameter ID and trial.
    When the framework ask tuner to generate hyper-parameters for a new trial,
    an ID has already been assigned and can be recorded in :meth:`generate_parameters`.
    Later when the trial ends, the ID will be reported to :meth:`trial_end`,
    and :meth:`receive_trial_result` if it has a final result.
    Parameter IDs are unique integers.

    The type/format of search space and hyper-parameters are not limited,
    as long as they are JSON-serializable and in sync with trial code.
    For HPO tuners, however, there is a widely shared common interface,
    which supports ``choice``, ``randint``, ``uniform``, and so on.
    See ``docs/en_US/Tutorial/SearchSpaceSpec.md`` for details of this interface.

    [WIP] For advanced tuners which take advantage of trials' intermediate results,
    an ``Advisor`` interface is under development.

    See Also
    --------
    Builtin tuners:
    :class:`~nni.algorithms.hpo.hyperopt_tuner.hyperopt_tuner.HyperoptTuner`
    :class:`~nni.algorithms.hpo.evolution_tuner.evolution_tuner.EvolutionTuner`
    :class:`~nni.algorithms.hpo.smac_tuner.SMACTuner`
    :class:`~nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner`
    :class:`~nni.algorithms.hpo.networkmorphism_tuner.networkmorphism_tuner.NetworkMorphismTuner`
    :class:`~nni.algorithms.hpo.metis_tuner.mets_tuner.MetisTuner`
    :class:`~nni.algorithms.hpo.ppo_tuner.PPOTuner`
    :class:`~nni.algorithms.hpo.gp_tuner.gp_tuner.GPTuner`
    """

    def generate_parameters(self, parameter_id: int, **kwargs) -> Parameters:
        if False:
            for i in range(10):
                print('nop')
        '\n        Abstract method which provides a set of hyper-parameters.\n\n        This method will get called when the framework is about to launch a new trial,\n        if user does not override :meth:`generate_multiple_parameters`.\n\n        The return value of this method will be received by trials via :func:`nni.get_next_parameter`.\n        It should fit in the search space, though the framework will not verify this.\n\n        User code must override either this method or :meth:`generate_multiple_parameters`.\n\n        Parameters\n        ----------\n        parameter_id : int\n            Unique identifier for requested hyper-parameters. This will later be used in :meth:`receive_trial_result`.\n        **kwargs\n            Unstable parameters which should be ignored by normal users.\n\n        Returns\n        -------\n        any\n            The hyper-parameters, a dict in most cases, but could be any JSON-serializable type when needed.\n\n        Raises\n        ------\n        nni.NoMoreTrialError\n            If the search space is fully explored, tuner can raise this exception.\n        '
        raise NotImplementedError('Tuner: generate_parameters not implemented')

    def generate_multiple_parameters(self, parameter_id_list: list[int], **kwargs) -> list[Parameters]:
        if False:
            while True:
                i = 10
        '\n        Callback method which provides multiple sets of hyper-parameters.\n\n        This method will get called when the framework is about to launch one or more new trials.\n\n        If user does not override this method, it will invoke :meth:`generate_parameters` on each parameter ID.\n\n        See :meth:`generate_parameters` for details.\n\n        User code must override either this method or :meth:`generate_parameters`.\n\n        Parameters\n        ----------\n        parameter_id_list : list of int\n            Unique identifiers for each set of requested hyper-parameters.\n            These will later be used in :meth:`receive_trial_result`.\n        **kwargs\n            Unstable parameters which should be ignored by normal users.\n\n        Returns\n        -------\n        list\n            List of hyper-parameters. An empty list indicates there are no more trials.\n        '
        result = []
        for parameter_id in parameter_id_list:
            try:
                _logger.debug('generating param for %s', parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                return result
            result.append(res)
        return result

    def receive_trial_result(self, parameter_id: int, parameters: Parameters, value: TrialMetric, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Abstract method invoked when a trial reports its final result. Must override.\n\n        This method only listens to results of algorithm-generated hyper-parameters.\n        Currently customized trials added from web UI will not report result to this method.\n\n        Parameters\n        ----------\n        parameter_id : int\n            Unique identifier of used hyper-parameters, same with :meth:`generate_parameters`.\n        parameters\n            Hyper-parameters generated by :meth:`generate_parameters`.\n        value\n            Result from trial (the return value of :func:`nni.report_final_result`).\n        **kwargs\n            Unstable parameters which should be ignored by normal users.\n        '
        raise NotImplementedError('Tuner: receive_trial_result not implemented')

    def _accept_customized_trials(self, accept=True):
        if False:
            for i in range(10):
                print('nop')
        self._accept_customized = accept

    def trial_end(self, parameter_id: int, success: bool, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Abstract method invoked when a trial is completed or terminated. Do nothing by default.\n\n        Parameters\n        ----------\n        parameter_id : int\n            Unique identifier for hyper-parameters used by this trial.\n        success : bool\n            True if the trial successfully completed; False if failed or terminated.\n        **kwargs\n            Unstable parameters which should be ignored by normal users.\n        '

    def update_search_space(self, search_space: SearchSpace) -> None:
        if False:
            return 10
        '\n        Abstract method for updating the search space. Must override.\n\n        Tuners are advised to support updating search space at run-time.\n        If a tuner can only set search space once before generating first hyper-parameters,\n        it should explicitly document this behaviour.\n\n        Parameters\n        ----------\n        search_space\n            JSON object defined by experiment owner.\n        '
        raise NotImplementedError('Tuner: update_search_space not implemented')

    def load_checkpoint(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Internal API under revising, not recommended for end users.\n        '
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Load checkpoint ignored by tuner, checkpoint path: %s', checkpoin_path)

    def save_checkpoint(self) -> None:
        if False:
            print('Hello World!')
        '\n        Internal API under revising, not recommended for end users.\n        '
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Save checkpoint ignored by tuner, checkpoint path: %s', checkpoin_path)

    def import_data(self, data: list[TrialRecord]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal API under revising, not recommended for end users.\n        '
        pass

    def _on_exit(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def _on_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass