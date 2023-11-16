from __future__ import annotations
import inspect
from river.stats import Var
from ..splitter import EBSTSplitter
from ..splitter.nominal_splitter_reg import NominalSplitterReg
from .leaf import HTLeaf

class LeafMean(HTLeaf):
    """Learning Node for regression tasks that always use the average target
        value as response.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if stats is None:
            stats = Var()
        super().__init__(stats, depth, splitter, **kwargs)

    @staticmethod
    def new_nominal_splitter():
        if False:
            print('Hello World!')
        return NominalSplitterReg()

    def manage_memory(self, criterion, last_check_ratio, last_check_vr, last_check_e):
        if False:
            i = 10
            return i + 15
        "Trigger Attribute Observers' memory management routines.\n\n        Currently, only `EBSTSplitter` and `TEBSTSplitter` have support to this feature.\n\n        Parameters\n        ----------\n        criterion\n            Split criterion\n        last_check_ratio\n            The ratio between the second best candidate's merit and the merit of the best\n            split candidate.\n        last_check_vr\n            The best candidate's split merit.\n        last_check_e\n            Hoeffding bound value calculated in the last split attempt.\n        "
        for splitter in self.splitters.values():
            if isinstance(splitter, EBSTSplitter):
                splitter.remove_bad_splits(criterion=criterion, last_check_ratio=last_check_ratio, last_check_vr=last_check_vr, last_check_e=last_check_e, pre_split_dist=self.stats)

    def update_stats(self, y, sample_weight):
        if False:
            for i in range(10):
                print('nop')
        self.stats.update(y, sample_weight)

    def prediction(self, x, *, tree=None):
        if False:
            i = 10
            return i + 15
        return self.stats.mean.get()

    @property
    def total_weight(self):
        if False:
            return 10
        'Calculate the total weight seen by the node.\n\n        Returns\n        -------\n        float\n            Total weight seen.\n\n        '
        return self.stats.mean.n

    def calculate_promise(self) -> int:
        if False:
            return 10
        'Estimate how likely a leaf node is going to be split.\n\n        Uses the node\'s depth as a heuristic to estimate how likely the leaf is going to become\n        a decision node. The deeper the node is in the tree, the more unlikely it is going to be\n        split. To cope with the general tree memory management framework, takes the negative of\n        the node\'s depth as return value. In this way, when sorting the tree leaves by their\n        "promise value", the deepest nodes are going to be placed at the first positions as\n        candidates to be deactivated.\n\n\n        Returns\n        -------\n        int\n            The smaller the value, the more unlikely the node is going to be split.\n\n        '
        return -self.depth

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{repr(self.stats.mean)} | {repr(self.stats)}' if self.stats else ''

class LeafModel(LeafMean):
    """Learning Node for regression tasks that always use a learning model to provide
        responses.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, leaf_model, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(stats, depth, splitter, **kwargs)
        self._leaf_model = leaf_model
        sign = inspect.signature(leaf_model.learn_one).parameters
        self._model_supports_weights = 'sample_weight' in sign or 'w' in sign

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        if False:
            for i in range(10):
                print('nop')
        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)
        if self._model_supports_weights:
            self._leaf_model.learn_one(x, y, sample_weight)
        else:
            for _ in range(int(sample_weight)):
                self._leaf_model.learn_one(x, y)

    def prediction(self, x, *, tree=None):
        if False:
            i = 10
            return i + 15
        return self._leaf_model.predict_one(x)

class LeafAdaptive(LeafModel):
    """Learning Node for regression tasks that dynamically selects between predictors and
        might behave as a regression tree node or a model tree node, depending on which predictor
        is the best one.

    Parameters
    ----------
    stats
        In regression tasks the node keeps an instance of `river.stats.Var` to estimate
        the target's statistics.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    leaf_model
        A `river.base.Regressor` instance used to learn from instances and provide
        responses.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, leaf_model, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(stats, depth, splitter, leaf_model, **kwargs)
        self._fmse_mean = 0.0
        self._fmse_model = 0.0

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None):
        if False:
            i = 10
            return i + 15
        pred_mean = self.stats.mean.get()
        pred_model = self._leaf_model.predict_one(x)
        self._fmse_mean = tree.model_selector_decay * self._fmse_mean + (y - pred_mean) ** 2
        self._fmse_model = tree.model_selector_decay * self._fmse_model + (y - pred_model) ** 2
        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

    def prediction(self, x, *, tree=None):
        if False:
            print('Hello World!')
        if self._fmse_mean < self._fmse_model:
            return self.stats.mean.get()
        else:
            return super().prediction(x)