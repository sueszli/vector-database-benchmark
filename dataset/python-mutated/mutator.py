from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Callable, Iterable, List, Optional, TYPE_CHECKING, Iterator
from numpy.random import RandomState
from nni.mutable import LabeledMutable, MutableList, MutableDict, Categorical, Mutable, SampleValidationError, Sample, SampleMissingError, label_scope, auto_label, frozen_context
from .space import ModelStatus
if TYPE_CHECKING:
    from .graph import GraphModelSpace
__all__ = ['MutationSampler', 'Mutator', 'StationaryMutator', 'InvalidMutation', 'MutatorSequence', 'Mutation']
Choice = Any

class MutationSampler:
    """
    Handles :meth:`Mutator.choice` calls.

    Choice is the only supported type for mutator.
    """

    def choice(self, candidates: List[Choice], mutator: 'Mutator', model: GraphModelSpace, index: int) -> Choice:
        if False:
            return 10
        raise NotImplementedError()

    def mutation_start(self, mutator: 'Mutator', model: GraphModelSpace) -> None:
        if False:
            print('Hello World!')
        pass

    def mutation_end(self, mutator: 'Mutator', model: GraphModelSpace) -> None:
        if False:
            while True:
                i = 10
        pass

class Mutator(LabeledMutable):
    """
    Mutates graphs in model to generate new model.

    By default, mutator simplifies to a single-value dict with its own label as key, and itself as value.
    At freeze, the strategy should provide a :class:`MutationSampler` in the dict.
    This is because the freezing of mutator is dynamic
    (i.e., requires a variational number of random numbers, dynamic ranges for each random number),
    and the :class:`MutationSampler` here can be considered as some random number generator
    to produce a random sequence based on the asks in :meth:`Mutator.mutate`.

    On the other hand, a subclass mutator should implement :meth:`Mutator.mutate`, which calls :meth:`Mutator.choice` inside,
    and :meth:`Mutator.choice` invokes the bounded sampler to "random" a choice.

    The label of the mutator in most cases is the label of the nodes on which the mutator is applied to.

    I imagine that mutating any model space (other than graph) might be useful,
    but we would postpone the support to when we actually need it.
    """

    def __init__(self, *, sampler: Optional[MutationSampler]=None, label: Optional[str]=None):
        if False:
            while True:
                i = 10
        self.sampler: Optional[MutationSampler] = sampler
        self.label: str = auto_label(label)
        self.model: Optional[GraphModelSpace] = None
        self._cur_model: Optional[GraphModelSpace] = None
        self._cur_choice_idx: Optional[int] = None

    def extra_repr(self) -> str:
        if False:
            return 10
        return f'label={self.label!r}'

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        if False:
            while True:
                i = 10
        'By default, treat self as a whole labeled mutable in the format dict.\n\n        Sub-class can override this to dry run the mutation upon the model and return the mutated model\n        for the followed-up dry run.\n\n        See Also\n        --------\n        nni.mutable.Mutable.leaf_mutables\n        '
        return super().leaf_mutables(is_leaf)

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        if False:
            return 10
        'Check if the sample is valid for this mutator.\n\n        See Also\n        --------\n        nni.mutable.Mutable.check_contains\n        '
        if self.label not in sample:
            return SampleMissingError(f'Mutator {self.label} not found in sample.')
        if not isinstance(sample[self.label], MutationSampler):
            return SampleValidationError(f'Mutator {self.label} is not a MutationSampler.')
        return None

    def freeze(self, sample: dict[str, Any]) -> GraphModelSpace:
        if False:
            i = 10
            return i + 15
        'When freezing a mutator, we need a model to mutate on, as well as a sampler to generate choices.\n\n        As how many times the mutator is applied on the model is often variational,\n        a sample with fixed length will not work.\n        The dict values in ``sample`` should be a sampler inheriting :class:`MutationSampler`.\n        But there are also cases where ``simplify()`` converts the mutation process into some fixed operations\n        (e.g., in :class:`StationaryMutator`).\n        In this case, sub-class should handle the freeze logic on their own.\n\n        :meth:`Mutator.freeze` needs to be called in a ``bind_model`` context.\n        '
        self.validate(sample)
        assert self.model is not None, 'Mutator must be bound to a model before freezing.'
        return self.bind_sampler(sample[self.label]).apply(self.model)

    def bind_sampler(self, sampler: MutationSampler) -> Mutator:
        if False:
            return 10
        'Set the sampler which will handle :meth:`Mutator.choice` calls.'
        self.sampler = sampler
        return self

    @contextmanager
    def bind_model(self, model: GraphModelSpace) -> Iterator[Mutator]:
        if False:
            for i in range(10):
                print('nop')
        'Mutators need a model, based on which they generate new models.\n        This context manager binds a model to the mutator, and unbinds it after the context.\n\n        Examples\n        --------\n        >>> with mutator.bind_model(model):\n        ...     mutator.simplify()\n        '
        try:
            self.model = model
            yield self
        finally:
            self.model = None

    def apply(self, model: GraphModelSpace) -> GraphModelSpace:
        if False:
            print('Hello World!')
        '\n        Apply this mutator on a model.\n        The model will be copied before mutation and the original model will not be modified.\n\n        Returns\n        -------\n        The mutated model.\n        '
        assert self.sampler is not None
        copy = model.fork()
        copy.status = ModelStatus.Mutating
        self._cur_model = copy
        self._cur_choice_idx = 0
        self._cur_samples = []
        copy.parent = Mutation(self, self._cur_samples, model, copy)
        self.sampler.mutation_start(self, copy)
        self.mutate(copy)
        self.sampler.mutation_end(self, copy)
        self._cur_model = None
        self._cur_choice_idx = None
        return copy

    def mutate(self, model: GraphModelSpace) -> None:
        if False:
            print('Hello World!')
        '\n        Abstract method to be implemented by subclass.\n        Mutate a model in place.\n        '
        raise NotImplementedError()

    def choice(self, candidates: Iterable[Choice]) -> Choice:
        if False:
            while True:
                i = 10
        'Ask sampler to make a choice.'
        assert self.sampler is not None and self._cur_model is not None and (self._cur_choice_idx is not None)
        ret = self.sampler.choice(list(candidates), self, self._cur_model, self._cur_choice_idx)
        self._cur_samples.append(ret)
        self._cur_choice_idx += 1
        return ret

    def random(self, memo: Sample | None=None, random_state: RandomState | None=None) -> GraphModelSpace | None:
        if False:
            i = 10
            return i + 15
        'Use a :class:`_RandomSampler` that generates a random sample when mutates.\n\n        See Also\n        --------\n        nni.mutable.Mutable.random\n        '
        sample: Sample = {} if memo is None else memo
        if random_state is None:
            random_state = RandomState()
        if self.label not in sample:
            sample[self.label] = _RandomSampler(random_state)
        if self.model is not None:
            return self.freeze(sample)
        else:
            return None

class StationaryMutator(Mutator):
    """A mutator that can be dry run.

    :class:`StationaryMutator` invoke :class:`StationaryMutator.dry_run` to predict choice candidates,
    such that the mutator simplifies to some static choices within `simplify()`.
    This could be convenient to certain algorithms which do not want to handle dynamic samplers.
    """

    def __init__(self, *, sampler: Optional[MutationSampler]=None, label: Optional[str]=None):
        if False:
            return 10
        super().__init__(sampler=sampler, label=label)
        self._dry_run_choices: Optional[MutableDict] = None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        if False:
            print('Hello World!')
        'Simplify this mutator to a number of static choices. Invokes :meth:`StationaryMutator.dry_run`.\n\n        Must be wrapped in a ``bind_model`` context.\n        '
        assert self.model is not None, 'Mutator must be bound to a model before calling `simplify()`.'
        (choices, model) = self.dry_run(self.model)
        self._dry_run_choices = MutableDict(choices)
        yield from self._dry_run_choices.leaf_mutables(is_leaf)
        self.model = model

    def check_contains(self, sample: dict[str, Any]):
        if False:
            return 10
        if self._dry_run_choices is None:
            raise RuntimeError('Dry run choices not found. Graph model space with stationary mutators must first invoke `simplify()` before freezing.')
        return self._dry_run_choices.check_contains(sample)

    def freeze(self, sample: dict[str, Any]) -> GraphModelSpace:
        if False:
            print('Hello World!')
        self.validate(sample)
        assert self._dry_run_choices is not None
        assert self.model is not None
        samples = [sample[label] for label in self._dry_run_choices]
        sampler = _FixedSampler(samples)
        return self.bind_sampler(sampler).apply(self.model)

    def dry_run(self, model: GraphModelSpace) -> tuple[dict[str, Categorical], GraphModelSpace]:
        if False:
            print('Hello World!')
        'Dry run mutator on a model to collect choice candidates.\n\n        If you invoke this method multiple times on same or different models,\n        it may or may not return identical results, depending on how the subclass implements `Mutator.mutate()`.\n\n        Recommended to be used in :meth:`simplify` if the mutator is static.\n        '
        sampler_backup = self.sampler
        recorder = _RecorderSampler()
        self.sampler = recorder
        new_model = self.apply(model)
        self.sampler = sampler_backup
        from nni.mutable.utils import label
        _label = label(self.label.split('/'))
        if len(recorder.recorded_candidates) != 1:
            with label_scope(_label):
                choices = [Categorical(candidates, label=str(i)) for (i, candidates) in enumerate(recorder.recorded_candidates)]
        else:
            choices = [Categorical(recorder.recorded_candidates[0], label=_label)]
        return ({c.label: c for c in choices}, new_model)

    def random(self, memo: Sample | None=None, random_state: RandomState | None=None) -> GraphModelSpace | None:
        if False:
            for i in range(10):
                print('nop')
        'Use :meth:`nni.mutable.Mutable.random` to generate a random sample.'
        return Mutable.random(self, memo, random_state)

class MutatorSequence(MutableList):
    """Apply a series of mutators on our model, sequentially.

    This could be generalized to a DAG indicating the dependencies between mutators,
    but we don't have a use case for that yet.
    """
    mutables: list[Mutator]

    def __init__(self, mutators: list[Mutator]):
        if False:
            while True:
                i = 10
        assert all((isinstance(mutator, Mutator) for mutator in mutators)), 'mutators must be a list of Mutator'
        super().__init__(mutators)
        self.model: Optional[GraphModelSpace] = None

    @contextmanager
    def bind_model(self, model: GraphModelSpace) -> Iterator[MutatorSequence]:
        if False:
            for i in range(10):
                print('nop')
        'Bind the model to a list of mutators.\n        The model (as well as its successors) will be bounded to the mutators one by one.\n        The model will be unbinded after the context.\n\n        Examples\n        --------\n        >>> with mutator_list.bind_model(model):\n        ...     mutator_list.freeze(samplers)\n        '
        try:
            self.model = model
            yield self
        finally:
            self.model = None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        if False:
            return 10
        assert self.model is not None, 'Mutator must be bound to a model before calling `simplify()`.'
        model = self.model
        with frozen_context():
            for mutator in self.mutables:
                with mutator.bind_model(model):
                    yield from mutator.leaf_mutables(is_leaf)
                    model = mutator.model
                    assert model is not None

    def freeze(self, sample: dict[str, Any]) -> GraphModelSpace:
        if False:
            for i in range(10):
                print('nop')
        assert self.model is not None, 'Mutator must be bound to a model before freezing.'
        model = self.model
        for mutator in self.mutables:
            with mutator.bind_model(model):
                model = mutator.freeze(sample)
        return model

class _RecorderSampler(MutationSampler):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.recorded_candidates: List[List[Choice]] = []

    def choice(self, candidates: List[Choice], *args) -> Choice:
        if False:
            return 10
        self.recorded_candidates.append(candidates)
        return candidates[0]

class _FixedSampler(MutationSampler):

    def __init__(self, samples):
        if False:
            while True:
                i = 10
        self.samples = samples

    def choice(self, candidates, mutator, model, index):
        if False:
            i = 10
            return i + 15
        if not 0 <= index < len(self.samples):
            raise RuntimeError(f'Invalid index {index} for samples {self.samples}')
        if self.samples[index] not in candidates:
            raise RuntimeError(f'Invalid sample {self.samples[index]} for candidates {candidates}')
        return self.samples[index]

class _RandomSampler(MutationSampler):

    def __init__(self, random_state: RandomState):
        if False:
            i = 10
            return i + 15
        self.random_state = random_state

    def choice(self, candidates, mutator, model, index):
        if False:
            while True:
                i = 10
        return self.random_state.choice(candidates)

class InvalidMutation(SampleValidationError):
    pass

class Mutation:
    """
    An execution of mutation, which consists of four parts: a mutator, a list of decisions (choices),
    the model that it comes from, and the model that it becomes.

    In general cases, the mutation logs are not reliable and should not be replayed as the mutators can
    be arbitrarily complex. However, for inline mutations, the labels correspond to mutator labels here,
    this can be useful for metadata visualization and python execution mode.

    Attributes
    ----------
    mutator
        Mutator.
    samples
        Decisions/choices.
    from_
        Model that is comes from.
    to
        Model that it becomes.
    """

    def __init__(self, mutator: 'Mutator', samples: List[Any], from_: GraphModelSpace, to: GraphModelSpace):
        if False:
            for i in range(10):
                print('nop')
        self.mutator: 'Mutator' = mutator
        self.samples: List[Any] = samples
        self.from_: GraphModelSpace = from_
        self.to: GraphModelSpace = to

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'Mutation(mutator={self.mutator}, samples={self.samples}, from={self.from_}, to={self.to})'