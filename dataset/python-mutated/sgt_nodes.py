from __future__ import annotations
import collections
import math
import numbers
import sys
from collections.abc import Hashable
from river import stats
from river.base.typing import FeatureName
from ..base import Leaf
from ..splitter.sgt_quantizer import DynamicQuantizer, StaticQuantizer
from ..utils import BranchFactory, GradHess, GradHessMerit, GradHessStats
from .branch import NominalMultiwayBranch, NumericBinaryBranch

class SGTLeaf(Leaf):
    """Leaf Node of the Stochastic Gradient Trees (SGT).

    There is only one type of leaf in SGTs. It handles only gradient and hessian
    information about the target. The tree handles target transformation and encoding.

    Parameters
    ----------
    prediction
        The initial prediction of the leaf.
    depth
        The depth of the leaf in the tree.
    split_params
        Parameters passed to the feature quantizers.
    """

    def __init__(self, prediction=0.0, depth=0, split_params=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._prediction = prediction
        self.depth = depth
        self.split_params = split_params if split_params is not None else collections.defaultdict(dict)
        self.last_split_attempt_at = 0
        self._split_stats: dict[FeatureName, dict[Hashable, GradHessStats] | DynamicQuantizer | StaticQuantizer] | None = {}
        self._update_stats = GradHessStats()

    def reset(self):
        if False:
            print('Hello World!')
        self._split_stats = {}
        self._update_stats = GradHessStats()

    @staticmethod
    def is_categorical(idx, x_val, nominal_attributes):
        if False:
            i = 10
            return i + 15
        return not isinstance(x_val, numbers.Number) or idx in nominal_attributes

    def update(self, x: dict, gh: GradHess, sgt, w: float=1.0):
        if False:
            for i in range(10):
                print('nop')
        for (idx, x_val) in x.items():
            if self.is_categorical(idx, x_val, sgt.nominal_attributes):
                sgt.nominal_attributes.add(idx)
                try:
                    self._split_stats[idx][x_val].update(gh, w)
                except KeyError:
                    if idx not in self._split_stats:
                        self._split_stats[idx] = {}
                    self._split_stats[idx][x_val] = GradHessStats()
                    self._split_stats[idx][x_val].update(gh, w)
            else:
                try:
                    self._split_stats[idx].update(x_val, gh, w)
                except KeyError:
                    self._split_stats[idx] = sgt.feature_quantizer.clone(self.split_params[idx])
                    self._split_stats[idx].update(x_val, gh, w)
        self._update_stats.update(gh, w=w)

    def prediction(self) -> float:
        if False:
            i = 10
            return i + 15
        return self._prediction

    def _eval_categorical_splits(self, feature_idx, candidate, sgt) -> tuple[BranchFactory, bool]:
        if False:
            i = 10
            return i + 15
        skip_candidate = True
        if feature_idx in sgt._split_features:
            return (candidate, skip_candidate)
        skip_candidate = False
        candidate.numerical_feature = False
        candidate.merit.delta_pred = {}
        all_dlms = stats.Var()
        cat_collection = self._split_stats[feature_idx]
        candidate.split_info = list(cat_collection.keys())
        for category in cat_collection:
            dp = self.delta_prediction(cat_collection[category].mean, sgt.lambda_value)
            dlms = cat_collection[category].delta_loss_mean_var(dp)
            candidate.merit.delta_pred[category] = dp
            all_dlms += dlms
        candidate.merit.loss_mean = all_dlms.mean.get() + len(cat_collection) * sgt.gamma / self.total_weight
        candidate.merit.loss_var = all_dlms.get()
        return (candidate, skip_candidate)

    def _eval_numerical_splits(self, feature_idx, candidate, sgt) -> tuple[BranchFactory, bool]:
        if False:
            print('Hello World!')
        skip_candidate = True
        quantizer = self._split_stats[feature_idx]
        self.split_params[feature_idx].update(quantizer._get_params())
        n_bins = len(quantizer)
        if n_bins == 1:
            return (candidate, skip_candidate)
        skip_candidate = False
        candidate.merit.loss_mean = math.inf
        candidate.merit.delta_pred = {}
        left_ghs = GradHessStats()
        left_dlms = stats.Var()
        for (thresh, ghs) in quantizer:
            left_ghs += ghs
            left_delta_pred = self.delta_prediction(left_ghs.mean, sgt.lambda_value)
            left_dlms += left_ghs.delta_loss_mean_var(left_delta_pred)
            right_ghs = self._update_stats - left_ghs
            right_delta_pred = self.delta_prediction(right_ghs.mean, sgt.lambda_value)
            right_dlms = right_ghs.delta_loss_mean_var(right_delta_pred)
            all_dlms = left_dlms + right_dlms
            loss_mean = all_dlms.mean.get()
            loss_var = all_dlms.get()
            if loss_mean < candidate.merit.loss_mean:
                candidate.merit.loss_mean = loss_mean + 2.0 * sgt.gamma / self.total_weight
                candidate.merit.loss_var = loss_var
                candidate.merit.delta_pred[0] = left_delta_pred
                candidate.merit.delta_pred[1] = right_delta_pred
                candidate.split_info = thresh
        return (candidate, skip_candidate)

    def find_best_split(self, sgt) -> BranchFactory:
        if False:
            return 10
        best_split = BranchFactory()
        best_split.merit = GradHessMerit()
        best_split.merit.delta_pred = self.delta_prediction(self._update_stats.mean, sgt.lambda_value)
        dlms = self._update_stats.delta_loss_mean_var(best_split.merit.delta_pred)
        best_split.merit.loss_mean = dlms.mean.get()
        best_split.merit.loss_var = dlms.get()
        for feature_idx in self._split_stats:
            candidate = BranchFactory()
            candidate.merit = GradHessMerit()
            candidate.feature = feature_idx
            if feature_idx in sgt.nominal_attributes:
                (candidate, skip_candidate) = self._eval_categorical_splits(feature_idx, candidate, sgt)
            else:
                (candidate, skip_candidate) = self._eval_numerical_splits(feature_idx, candidate, sgt)
            if skip_candidate:
                continue
            if candidate.merit.loss_mean < best_split.merit.loss_mean:
                best_split = candidate
        return best_split

    def apply_split(self, split, p_node, p_branch, sgt):
        if False:
            return 10
        if split.feature is None:
            self._prediction += split.merit.delta_pred
            sgt._n_node_updates += 1
            self.reset()
            return
        sgt._n_splits += 1
        sgt._split_features.add(split.feature)
        branch = NumericBinaryBranch if split.numerical_feature else NominalMultiwayBranch
        child_depth = self.depth + 1
        leaves = tuple((SGTLeaf(self._prediction + delta_pred, depth=child_depth, split_params=self.split_params.copy()) for delta_pred in split.merit.delta_pred.values()))
        new_split = split.assemble(branch, self.split_params.copy(), self.depth, *leaves)
        if p_branch is None:
            sgt._root = new_split
        else:
            p_node.children[p_branch] = new_split

    @property
    def total_weight(self) -> float:
        if False:
            i = 10
            return i + 15
        return self._update_stats.total_weight

    @property
    def update_stats(self):
        if False:
            while True:
                i = 10
        return self._update_stats

    @staticmethod
    def delta_prediction(gh: GradHess, lambda_value: float):
        if False:
            while True:
                i = 10
        return -gh.gradient / (gh.hessian + sys.float_info.min + lambda_value)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self.prediction())