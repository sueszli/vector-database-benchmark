import h2o

class FeatureInteraction:

    def _feature_interaction(self, max_interaction_depth=100, max_tree_depth=100, max_deepening=-1, path=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Feature interactions and importance, leaf statistics and split value histograms in a tabular form.\n        Available for XGBoost and GBM.\n\n        Metrics:\n        Gain - Total gain of each feature or feature interaction.\n        FScore - Amount of possible splits taken on a feature or feature interaction.\n        wFScore - Amount of possible splits taken on a feature or feature interaction weighed by \n        the probability of the splits to take place.\n        Average wFScore - wFScore divided by FScore.\n        Average Gain - Gain divided by FScore.\n        Expected Gain - Total gain of each feature or feature interaction weighed by the probability to gather the gain.\n        Average Tree Index\n        Average Tree Depth\n\n        :param max_interaction_depth: Upper bound for extracted feature interactions depth. Defaults to 100.\n        :param max_tree_depth: Upper bound for tree depth. Defaults to 100.\n        :param max_deepening: Upper bound for interaction start deepening (zero deepening => interactions \n            starting at root only). Defaults to -1.\n        :param path: (Optional) Path where to save the output in .xlsx format (e.g. ``/mypath/file.xlsx``).\n            Please note that Pandas and XlsxWriter need to be installed for using this option. Defaults to None.\n\n\n        :examples:\n        >>> boston = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/BostonHousing.csv")\n        >>> predictors = boston.columns[:-1]\n        >>> response = "medv"\n        >>> boston[\'chas\'] = boston[\'chas\'].asfactor()\n        >>> train, valid = boston.split_frame(ratios=[.8])\n        >>> boston_xgb = H2OXGBoostEstimator(seed=1234)\n        >>> boston_xgb.train(y=response, x=predictors, training_frame=train)\n        >>> feature_interactions = boston_xgb.feature_interaction()\n        '
        kwargs = {}
        kwargs['model_id'] = self.model_id
        kwargs['max_interaction_depth'] = max_interaction_depth
        kwargs['max_tree_depth'] = max_tree_depth
        kwargs['max_deepening'] = max_deepening
        json = h2o.api('POST /3/FeatureInteraction', data=kwargs)
        if path is not None:
            import pandas as pd
            writer = pd.ExcelWriter(path, engine='xlsxwriter')
            for fi in json['feature_interaction']:
                fi.as_data_frame().to_excel(writer, sheet_name=fi._table_header)
            writer.save()
        return json['feature_interaction']