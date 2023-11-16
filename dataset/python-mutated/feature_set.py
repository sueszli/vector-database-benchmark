import itertools
import logging
from collections import defaultdict
from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base import AggregationFeature, FeatureOutputSlice, GroupByTransformFeature, TransformFeature
from featuretools.utils import Trie
logger = logging.getLogger('featuretools.computational_backend')

class FeatureSet(object):
    """
    Represents an immutable set of features to be calculated for a single dataframe, and their
    dependencies.
    """

    def __init__(self, features, approximate_feature_trie=None):
        if False:
            return 10
        '\n        Args:\n            features (list[Feature]): Features of the target dataframe.\n            approximate_feature_trie (Trie[RelationshipPath, set[str]], optional): Dependency\n                features to ignore because they have already been approximated. For example, if\n                one of the target features is a direct feature of a feature A and A is included in\n                approximate_feature_trie then neither A nor its dependencies will appear in\n                FeatureSet.feature_trie.\n        '
        self.target_df_name = features[0].dataframe_name
        self.target_features = features
        self.target_feature_names = {f.unique_name() for f in features}
        if not approximate_feature_trie:
            approximate_feature_trie = Trie(default=list, path_constructor=RelationshipPath)
        self.approximate_feature_trie = approximate_feature_trie
        self.features_by_name = {f.unique_name(): f for f in features}
        feature_dependents = defaultdict(set)
        for f in features:
            deps = f.get_dependencies(deep=True)
            for dep in deps:
                feature_dependents[dep.unique_name()].add(f.unique_name())
                self.features_by_name[dep.unique_name()] = dep
                subdeps = dep.get_dependencies(deep=True)
                for sd in subdeps:
                    feature_dependents[sd.unique_name()].add(dep.unique_name())
        self.feature_dependents = {fname: [self.features_by_name[dname] for dname in feature_dependents[fname]] for (fname, f) in self.features_by_name.items()}
        self._feature_trie = None

    @property
    def feature_trie(self):
        if False:
            return 10
        '\n        The target features and their dependencies organized into a trie by relationship path.\n        This is built once when it is first called (to avoid building it if it is not needed) and\n        then used for all subsequent calls.\n\n        The edges of the trie are RelationshipPaths and the values are tuples of\n        (bool, set[str], set[str]). The bool represents whether the full dataframe is needed at\n        that node, the first set contains the names of features which are needed on the full\n        dataframe, and the second set contains the names of the rest of the features\n\n        Returns:\n            Trie[RelationshipPath, (bool, set[str], set[str])]\n        '
        if not self._feature_trie:
            self._feature_trie = self._build_feature_trie()
        return self._feature_trie

    def _build_feature_trie(self):
        if False:
            return 10
        '\n        Build the feature trie by adding the target features and their dependencies recursively.\n        '
        feature_trie = Trie(default=lambda : (False, set(), set()), path_constructor=RelationshipPath)
        for f in self.target_features:
            self._add_feature_to_trie(feature_trie, f, self.approximate_feature_trie)
        return feature_trie

    def _add_feature_to_trie(self, trie, feature, approximate_feature_trie, ancestor_needs_full_dataframe=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add the given feature to the root of the trie, and recurse on its dependencies. If it is in\n        approximate_feature_trie then it will not be added and we will not recurse on its dependencies.\n        '
        (node_needs_full_dataframe, full_features, not_full_features) = trie.value
        needs_full_dataframe = ancestor_needs_full_dataframe or self.uses_full_dataframe(feature)
        name = feature.unique_name()
        if name in approximate_feature_trie.value:
            return
        if needs_full_dataframe:
            full_features.add(name)
            if name in not_full_features:
                not_full_features.remove(name)
            trie.value = (True, full_features, not_full_features)
            sub_trie = trie
            for edge in feature.relationship_path:
                sub_trie = sub_trie.get_node([edge])
                (_, f1, f2) = sub_trie.value
                sub_trie.value = (True, f1, f2)
        else:
            if name not in full_features:
                not_full_features.add(name)
            sub_trie = trie.get_node(feature.relationship_path)
        sub_ignored_trie = approximate_feature_trie.get_node(feature.relationship_path)
        for dep_feat in feature.get_dependencies():
            if isinstance(dep_feat, FeatureOutputSlice):
                dep_feat = dep_feat.base_feature
            self._add_feature_to_trie(sub_trie, dep_feat, sub_ignored_trie, ancestor_needs_full_dataframe=needs_full_dataframe)

    def group_features(self, feature_names):
        if False:
            print('Hello World!')
        '\n        Topologically sort the given features, then group by path,\n        feature type, use_previous, and where.\n        '
        features = [self.features_by_name[name] for name in feature_names]
        depths = self._get_feature_depths(features)

        def key_func(f):
            if False:
                while True:
                    i = 10
            return (depths[f.unique_name()], f.relationship_path_name(), str(f.__class__), _get_use_previous(f), _get_where(f), self.uses_full_dataframe(f), _get_groupby(f))
        sort_feats = sorted(features, key=key_func)
        feature_groups = [list(g) for (_, g) in itertools.groupby(sort_feats, key=key_func)]
        return feature_groups

    def _get_feature_depths(self, features):
        if False:
            while True:
                i = 10
        '\n        Generate and return a mapping of {feature name -> depth} in the\n        feature DAG for the given dataframe.\n        '
        order = defaultdict(int)
        depths = {}
        queue = features[:]
        while queue:
            f = queue.pop(0)
            depths[f.unique_name()] = order[f.unique_name()]
            if not f.relationship_path:
                dependencies = f.get_dependencies()
                for dep in dependencies:
                    order[dep.unique_name()] = min(order[f.unique_name()] - 1, order[dep.unique_name()])
                    queue.append(dep)
        return depths

    def uses_full_dataframe(self, feature, check_dependents=False):
        if False:
            i = 10
            return i + 15
        if isinstance(feature, TransformFeature) and feature.primitive.uses_full_dataframe:
            return True
        return check_dependents and self._dependent_uses_full_dataframe(feature)

    def _dependent_uses_full_dataframe(self, feature):
        if False:
            return 10
        for d in self.feature_dependents[feature.unique_name()]:
            if isinstance(d, TransformFeature) and d.primitive.uses_full_dataframe:
                return True
        return False

def _get_use_previous(f):
    if False:
        while True:
            i = 10
    if isinstance(f, AggregationFeature) and f.use_previous is not None:
        if len(f.use_previous.times.keys()) > 1:
            return ('', -1)
        else:
            unit = list(f.use_previous.times.keys())[0]
            value = f.use_previous.times[unit]
            return (unit, value)
    else:
        return ('', -1)

def _get_where(f):
    if False:
        while True:
            i = 10
    if isinstance(f, AggregationFeature) and f.where is not None:
        return f.where.unique_name()
    else:
        return ''

def _get_groupby(f):
    if False:
        i = 10
        return i + 15
    if isinstance(f, GroupByTransformFeature):
        return f.groupby.unique_name()
    else:
        return ''