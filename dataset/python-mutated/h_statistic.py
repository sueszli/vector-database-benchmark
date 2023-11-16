import h2o

class HStatistic:

    def _h(self, frame, variables):
        if False:
            return 10
        '\n        Calculates Friedman and Popescu\'s H statistics, in order to test for the presence of an interaction between specified variables in h2o gbm and xgb models.\n        H varies from 0 to 1. It will have a value of 0 if the model exhibits no interaction between specified variables and a correspondingly larger value for a \n        stronger interaction effect between them. NaN is returned if a computation is spoiled by weak main effects and rounding errors.\n        \n        This statistic can be calculated only for numerical variables. Missing values are supported.\n        \n        See Jerome H. Friedman and Bogdan E. Popescu, 2008, "Predictive learning via rule ensembles", *Ann. Appl. Stat.*\n        **2**:916-954, http://projecteuclid.org/download/pdfview_1/euclid.aoas/1223908046, s. 8.1.\n\n        \n        :param frame: the frame that current model has been fitted to\n        :param variables: variables of the interest\n        :return: H statistic of the variables \n        \n        :examples:\n        >>> prostate_train = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/logreg/prostate_train.csv")\n        >>> prostate_train["CAPSULE"] = prostate_train["CAPSULE"].asfactor()\n        >>> gbm_h2o = H2OGradientBoostingEstimator(ntrees=100, learn_rate=0.1,\n        >>>                                 max_depth=5,\n        >>>                                 min_rows=10,\n        >>>                                 distribution="bernoulli")\n        >>> gbm_h2o.train(x=list(range(1,prostate_train.ncol)),y="CAPSULE", training_frame=prostate_train)\n        >>> h = gbm_h2o.h(prostate_train, [\'DPROS\',\'DCAPS\'])\n        '
        kwargs = dict(model_id=self.model_id, frame=frame.key, variables=variables)
        json = h2o.api('POST /3/FriedmansPopescusH', data=kwargs)
        return json['h']