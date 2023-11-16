from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as tc
from turicreate.toolkits._internal_utils import _raise_error_if_not_sframe
from turicreate.toolkits._supervised_learning import select_default_missing_value_policy
from turicreate.toolkits import _coreml_utils

class TreeModelMixin(object):
    """
    Implements common methods among tree models:
    - BoostedTreesClassifier
    - BoostedTreesRegression
    - RandomForestClassifier
    - RandomForestRegression
    - DecisionTreeClassifier
    - DecisionTreeRegression
    """

    def get_feature_importance(self):
        if False:
            while True:
                i = 10
        '\n        Get the importance of features used by the model.\n\n        The measure of importance of feature X\n        is determined by the sum of occurrence of X\n        as a branching node in all trees.\n\n        When X is a categorical feature, e.g. "Gender",\n        the index column contains the value of the feature, e.g. "M" or "F".\n        When X is a numerical feature, index of X is None.\n\n        Returns\n        -------\n        out : SFrame\n            A table with three columns: name, index, count,\n            ordered by \'count\' in descending order.\n\n        Examples\n        --------\n        >>> m.get_feature_importance()\n        Rows: 31\n        Data:\n            +-----------------------------+-------+-------+\n            |             name            | index | count |\n            +-----------------------------+-------+-------+\n            | DER_mass_transverse_met_lep |  None |   66  |\n            |         DER_mass_vis        |  None |   65  |\n            |          PRI_tau_pt         |  None |   61  |\n            |         DER_mass_MMC        |  None |   59  |\n            |      DER_deltar_tau_lep     |  None |   58  |\n            |          DER_pt_tot         |  None |   41  |\n            |           PRI_met           |  None |   38  |\n            |     PRI_jet_leading_eta     |  None |   30  |\n            |     DER_deltaeta_jet_jet    |  None |   27  |\n            |       DER_mass_jet_jet      |  None |   24  |\n            +-----------------------------+-------+-------+\n            [31 rows x 3 columns]\n        '
        return tc.extensions._xgboost_feature_importance(self.__proxy__)

    def extract_features(self, dataset, missing_value_action='auto'):
        if False:
            return 10
        "\n        For each example in the dataset, extract the leaf indices of\n        each tree as features.\n\n        For multiclass classification, each leaf index contains #num_class\n        numbers.\n\n        The returned feature vectors can be used as input to train another\n        supervised learning model such as a\n        :py:class:`~turicreate.logistic_classifier.LogisticClassifier`,\n        or a :py:class:`~turicreate.svm_classifier.SVMClassifier`.\n\n        Parameters\n        ----------\n        dataset : SFrame\n            Dataset of new observations. Must include columns with the same\n            names as the features used for model training, but does not require\n            a target column. Additional columns are ignored.\n\n        missing_value_action: str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Choose a model dependent missing value policy.\n            - 'impute': Proceed with evaluation by filling in the missing\n                        values with the mean of the training data. Missing\n                        values are also imputed if an entire column of data is\n                        missing during evaluation.\n            - 'none': Treat missing value as is. Model must be able to handle\n                      missing value.\n            - 'error' : Do not proceed with prediction and terminate with\n                        an error message.\n\n        Returns\n        -------\n        out : SArray\n            An SArray of dtype array.array containing extracted features.\n\n        Examples\n        --------\n        >>> data =  turicreate.SFrame(\n            'https://static.turi.com/datasets/regression/houses.csv')\n\n        >>> # Regression Tree Models\n        >>> data['regression_tree_features'] = model.extract_features(data)\n\n        >>> # Classification Tree Models\n        >>> data['classification_tree_features'] = model.extract_features(data)\n        "
        _raise_error_if_not_sframe(dataset, 'dataset')
        if missing_value_action == 'auto':
            missing_value_action = select_default_missing_value_policy(self, 'extract_features')
        return self.__proxy__.extract_features(dataset, missing_value_action)

    def _extract_features_with_missing(self, dataset, tree_id=0, missing_value_action='auto'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Extract features along with all the missing features associated with\n        a dataset.\n\n        Parameters\n        ----------\n        dataset: bool\n            Dataset on which to make predictions.\n\n        missing_value_action: str, optional\n            Action to perform when missing values are encountered. This can be\n            one of:\n\n            - 'auto': Choose a model dependent missing value policy.\n            - 'impute': Proceed with evaluation by filling in the missing\n                        values with the mean of the training data. Missing\n                        values are also imputed if an entire column of data is\n                        missing during evaluation.\n            - 'none': Treat missing value as is. Model must be able to handle\n                      missing value.\n            - 'error' : Do not proceed with prediction and terminate with\n                        an error message.\n\n        Returns\n        -------\n        out : SFrame\n            A table with two columns:\n              - leaf_id          : Leaf id of the corresponding tree.\n              - missing_features : A list of missing feature, index pairs\n        "
        sf = dataset
        sf['leaf_id'] = self.extract_features(dataset, missing_value_action).vector_slice(tree_id).astype(int)
        tree = self._get_tree(tree_id)
        type_map = dict(zip(dataset.column_names(), dataset.column_types()))

        def get_missing_features(row):
            if False:
                i = 10
                return i + 15
            x = row['leaf_id']
            path = tree.get_prediction_path(x)
            missing_id = []
            for p in path:
                fname = p['feature']
                idx = p['index']
                f = row[fname]
                if type_map[fname] in [int, float]:
                    if f is None:
                        missing_id.append(p['child_id'])
                elif type_map[fname] in [dict]:
                    if f is None:
                        missing_id.append(p['child_id'])
                    if idx not in f:
                        missing_id.append(p['child_id'])
                else:
                    pass
            return missing_id
        sf['missing_id'] = sf.apply(get_missing_features, list)
        return sf[['leaf_id', 'missing_id']]

    def _dump_to_text(self, with_stats):
        if False:
            while True:
                i = 10
        "\n        Dump the models into a list of strings. Each\n        string is a text representation of a tree.\n\n        Parameters\n        ----------\n        with_stats : bool\n            If true, include node statistics in the output.\n\n        Returns\n        -------\n        out : SFrame\n            A table with two columns: feature, count,\n            ordered by 'count' in descending order.\n        "
        return tc.extensions._xgboost_dump_model(self.__proxy__, with_stats=with_stats, format='text')

    def _dump_to_json(self, with_stats):
        if False:
            print('Hello World!')
        "\n        Dump the models into a list of strings. Each\n        string is a text representation of a tree.\n\n        Parameters\n        ----------\n        with_stats : bool\n            If true, include node statistics in the output.\n\n        Returns\n        -------\n        out : SFrame\n            A table with two columns: feature, count,\n            ordered by 'count' in descending order.\n        "
        import json
        trees_json_str = tc.extensions._xgboost_dump_model(self.__proxy__, with_stats=with_stats, format='json')
        trees_json = [json.loads(x) for x in trees_json_str]
        import struct
        import sys

        def hexadecimal_to_float(s):
            if False:
                for i in range(10):
                    print('nop')
            if sys.version_info[0] >= 3:
                return struct.unpack('<f', bytes.fromhex(s))[0]
            else:
                return struct.unpack('<f', s.decode('hex'))[0]
        for d in trees_json:
            nodes = d['vertices']
            for n in nodes:
                if 'value_hexadecimal' in n:
                    n['value'] = hexadecimal_to_float(n['value_hexadecimal'])
        return trees_json

    def _get_tree(self, tree_id=0):
        if False:
            for i in range(10):
                print('nop')
        "\n        A simple pure python wrapper around a single (or ensemble) of Turi\n        create decision trees.\n\n        The tree can be obtained directly from the `trees_json` parameter in\n        any GLC tree model objects (boosted trees, random forests, and decision\n        trees).\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            # Train a tree ensemble.\n            >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'\n            >>> data = tc.SFrame.read_csv(url)\n\n            >>> train, test = data.random_split(0.8)\n            >>> model = tc.boosted_trees_classifier.create(train,\n            ...                  target='label', validation_set = None)\n\n            # Obtain the model as a tree object.\n            >>> tree = model._get_tree()\n\n            >>> tree = DecisionTree(model, tree_id = 0)\n\n        "
        from . import _decision_tree
        return _decision_tree.DecisionTree.from_model(self, tree_id)

    def _get_summary_struct(self):
        if False:
            return 10
        "\n        Returns a structured description of the model, including (where relevant)\n        the schema of the training data, description of the training data,\n        training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        data_fields = [('Number of examples', 'num_examples'), ('Number of feature columns', 'num_features'), ('Number of unpacked features', 'num_unpacked_features')]
        if 'num_classes' in self._list_fields():
            data_fields.append(('Number of classes', 'num_classes'))
        training_fields = [('Number of trees', 'num_trees'), ('Max tree depth', 'max_depth'), ('Training time (sec)', 'training_time')]
        for m in ['accuracy', 'log_loss', 'auc', 'rmse', 'max_error']:
            if 'training_%s' % m in self._list_fields():
                training_fields.append(('Training %s' % m, 'training_%s' % m))
                if 'validation_%s' % m in self._list_fields():
                    training_fields.append(('Validation %s' % m, 'validation_%s' % m))
        return ([data_fields, training_fields], ['Schema', 'Settings'])

    def _export_coreml_impl(self, filename, context):
        if False:
            while True:
                i = 10
        info = _coreml_utils._get_model_metadata(context['class'], None)
        if 'user_defined' not in context:
            context['user_defined'] = info
        else:
            context['user_defined'].update(info)
        tc.extensions._xgboost_export_as_model_asset(self.__proxy__, filename, context)