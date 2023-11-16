import pandas as pd
from pycaret.internal.pycaret_experiment.supervised_experiment import _SupervisedExperiment
from pycaret.utils.time_series.forecasting.pipeline import _pipeline_transform

class _TSSupervisedExperiment(_SupervisedExperiment):

    @property
    def X(self):
        if False:
            i = 10
            return i + 15
        X = self.dataset.drop(self.target_param, axis=1)
        if X.empty and self.fe_exogenous is None:
            return None
        else:
            return X

    @property
    def dataset_transformed(self):
        if False:
            i = 10
            return i + 15
        return pd.concat([*_pipeline_transform(pipeline=self.pipeline_fully_trained, y=self.y, X=self.X)], axis=1)

    @property
    def X_train_transformed(self):
        if False:
            print('Hello World!')
        return _pipeline_transform(pipeline=self.pipeline, y=self.y_train, X=self.X_train)[1]

    @property
    def train_transformed(self):
        if False:
            i = 10
            return i + 15
        return pd.concat([*_pipeline_transform(pipeline=self.pipeline, y=self.y_train, X=self.X_train)], axis=1)

    @property
    def X_transformed(self):
        if False:
            print('Hello World!')
        return _pipeline_transform(pipeline=self.pipeline_fully_trained, y=self.y, X=self.X)[1]

    @property
    def X_train(self):
        if False:
            i = 10
            return i + 15
        X_train = self.train.drop(self.target_param, axis=1)
        if X_train.empty and self.fe_exogenous is None:
            return None
        else:
            return X_train

    @property
    def X_test(self):
        if False:
            for i in range(10):
                print('nop')
        test = self.dataset.loc[self.idx[2], :]
        X_test = test.drop(self.target_param, axis=1)
        if X_test.empty and self.fe_exogenous is None:
            return None
        else:
            return X_test

    @property
    def test(self):
        if False:
            return 10
        return self.dataset.loc[self.idx[1], :]

    @property
    def test_transformed(self):
        if False:
            print('Hello World!')
        all_data = pd.concat([*_pipeline_transform(pipeline=self.pipeline_fully_trained, y=self.y, X=self.X)], axis=1)
        return all_data.loc[self.idx[1]]

    @property
    def y_transformed(self):
        if False:
            i = 10
            return i + 15
        return _pipeline_transform(pipeline=self.pipeline_fully_trained, y=self.y, X=self.X)[0]

    @property
    def X_test_transformed(self):
        if False:
            return 10
        (_, X) = _pipeline_transform(pipeline=self.pipeline_fully_trained, y=self.y, X=self.X)
        if X is None:
            return None
        else:
            return X.loc[self.idx[2]]

    @property
    def y_train_transformed(self):
        if False:
            print('Hello World!')
        return _pipeline_transform(pipeline=self.pipeline, y=self.y_train, X=self.X_train)[0]

    @property
    def y_test_transformed(self):
        if False:
            while True:
                i = 10
        (y, _) = _pipeline_transform(pipeline=self.pipeline_fully_trained, y=self.y, X=self.X)
        return y.loc[self.idx[1]]

    def _create_model_get_train_X_y(self, X_train, y_train):
        if False:
            print('Hello World!')
        'Return appropriate training X and y values depending on whether\n        X_train and y_train are passed or not. If X_train and y_train are not\n        passes, internal self.X_train and self.y_train are returned. If they are\n        passed, then a copy of them is returned.'
        data_X = self.X_train if X_train is None else X_train.copy()
        data_y = self.y_train if y_train is None else y_train.copy()
        return (data_X, data_y)