import pandas as pd
from pycaret.internal.pycaret_experiment.supervised_experiment import _SupervisedExperiment

class _NonTSSupervisedExperiment(_SupervisedExperiment):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    @property
    def test(self):
        if False:
            i = 10
            return i + 15
        'Test set.'
        return self.dataset.loc[self.idx[1], :]

    @property
    def X(self):
        if False:
            return 10
        'Feature set.'
        return self.dataset.drop(self.target_param, axis=1)

    @property
    def X_train(self):
        if False:
            print('Hello World!')
        'Feature set of the training set.'
        return self.train.drop(self.target_param, axis=1)

    @property
    def X_test(self):
        if False:
            for i in range(10):
                print('nop')
        'Feature set of the test set.'
        return self.test.drop(self.target_param, axis=1)

    @property
    def dataset_transformed(self):
        if False:
            return 10
        'Transformed dataset.'
        return pd.concat([self.train_transformed, self.test_transformed])

    @property
    def train_transformed(self):
        if False:
            while True:
                i = 10
        'Transformed training set.'
        return pd.concat([*self.pipeline.transform(X=self.X_train, y=self.y_train, filter_train_only=False)], axis=1)

    @property
    def test_transformed(self):
        if False:
            return 10
        'Transformed test set.'
        return pd.concat([*self.pipeline.transform(X=self.X_test, y=self.y_test)], axis=1)

    @property
    def X_transformed(self):
        if False:
            return 10
        'Transformed feature set.'
        return pd.concat([self.X_train_transformed, self.X_test_transformed])

    @property
    def y_transformed(self):
        if False:
            for i in range(10):
                print('nop')
        'Transformed target column.'
        return pd.concat([self.y_train_transformed, self.y_test_transformed])

    @property
    def X_train_transformed(self):
        if False:
            i = 10
            return i + 15
        'Transformed feature set of the training set.'
        return self.train_transformed.drop(self.target_param, axis=1)

    @property
    def y_train_transformed(self):
        if False:
            print('Hello World!')
        'Transformed target column of the training set.'
        return self.train_transformed[self.target_param]

    @property
    def X_test_transformed(self):
        if False:
            for i in range(10):
                print('nop')
        'Transformed feature set of the test set.'
        return self.test_transformed.drop(self.target_param, axis=1)

    @property
    def y_test_transformed(self):
        if False:
            for i in range(10):
                print('nop')
        'Transformed target column of the test set.'
        return self.test_transformed[self.target_param]

    def _create_model_get_train_X_y(self, X_train, y_train):
        if False:
            print('Hello World!')
        'Return appropriate training X and y values depending on whether\n        X_train and y_train are passed or not. If X_train and y_train are not\n        passes, internal self.X_train and self.y_train are returned. If they are\n        passed, then a copy of them is returned.\n        '
        if X_train is not None:
            data_X = X_train.copy()
        elif self.X_train is None:
            data_X = None
        else:
            data_X = self.X_train
        data_y = self.y_train if y_train is None else y_train.copy()
        return (data_X, data_y)