"""
This module implements attribute inference attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Union, Tuple, List, Any, TYPE_CHECKING
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import minmax_scale, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.regression import RegressorMixin
from art.attacks.attack import AttributeInferenceAttack
from art.utils import check_and_transform_label_format, float_to_categorical, floats_to_one_hot, get_feature_values, remove_attacked_feature
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, REGRESSOR_TYPE
logger = logging.getLogger(__name__)

class AttributeInferenceBaselineTrueLabel(AttributeInferenceAttack):
    """
    Implementation of a baseline attribute inference, not using a model.

    The idea is to train a simple neural network to learn the attacked feature from the rest of the features, and the
    true label. Should be used to compare with other attribute inference results.
    """
    attack_params = AttributeInferenceAttack.attack_params + ['prediction_normal_factor', 'scale_range', 'attack_model_type', 'is_continuous', 'is_regression', 'non_numerical_features', 'encoder']
    _estimator_requirements = ()

    def __init__(self, attack_model_type: str='nn', attack_model: Optional[Union['CLASSIFIER_TYPE', 'REGRESSOR_TYPE']]=None, attack_feature: Union[int, slice]=0, is_continuous: Optional[bool]=False, is_regression: Optional[bool]=False, scale_range: Optional[Tuple[float, float]]=None, prediction_normal_factor: float=1, non_numerical_features: Optional[List[int]]=None, encoder: Optional[Union[OrdinalEncoder, OneHotEncoder, ColumnTransformer]]=None, nn_model_epochs: int=100, nn_model_batch_size: int=100, nn_model_learning_rate: float=0.0001):
        if False:
            while True:
                i = 10
        "\n        Create an AttributeInferenceBaseline attack instance.\n\n        :param attack_model_type: the type of default attack model to train, optional. Should be one of:\n                                 `nn` (neural network, default),\n                                 `rf` (random forest),\n                                 `gb` (gradient boosting),\n                                 `lr` (logistic/linear regression),\n                                 `dt` (decision tree),\n                                 `knn` (k nearest neighbors),\n                                 `svm` (support vector machine).\n                                  If `attack_model` is supplied, this option will be ignored.\n        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.\n        :param attack_feature: The index of the feature to be attacked or a slice representing multiple indexes in\n                               case of a one-hot encoded feature.\n        :param is_continuous: Whether the attacked feature is continuous. Default is False (which means categorical).\n        :param is_regression: Whether the model is a regression model. Default is False (classification).\n        :param scale_range: If supplied, the class labels (both true and predicted) will be scaled to the given range.\n                            Only applicable when `is_regression` is True.\n        :param prediction_normal_factor: If supplied, the class labels (both true and predicted) are multiplied by the\n                                         factor when used as inputs to the attack-model. Only applicable when\n                                         `is_regression` is True and if `scale_range` is not supplied.\n        :param non_numerical_features: a list of feature indexes that require encoding in order to feed into an ML model\n                                       (i.e., strings), not including the attacked feature. Should only be supplied if\n                                       non-numeric features exist in the input data not including the attacked feature,\n                                       and an encoder is not supplied.\n        :param encoder: An already fit encoder that can be applied to the model's input features without the attacked\n                        feature (i.e., should be fit for n-1 features).\n        :param nn_model_epochs: the number of epochs to use when training a nn attack model\n        :param nn_model_batch_size: the batch size to use when training a nn attack model\n        :param nn_model_learning_rate: the learning rate to use when training a nn attack model\n        "
        super().__init__(estimator=None, attack_feature=attack_feature)
        self._values: list = []
        self._encoder = encoder
        self._non_numerical_features = non_numerical_features
        self._is_continuous = is_continuous
        self._attack_model_type: Optional[str] = attack_model_type
        self.attack_model: Optional[Any] = None
        self.epochs = nn_model_epochs
        self.batch_size = nn_model_batch_size
        self.learning_rate = nn_model_learning_rate
        if attack_model:
            if self._is_continuous:
                if RegressorMixin not in type(attack_model).__mro__:
                    raise ValueError('When attacking a continuous feature the attack model must be of type Regressor.')
            elif ClassifierMixin not in type(attack_model).__mro__:
                raise ValueError('When attacking a categorical feature the attack model must be of type Classifier.')
            self.attack_model = attack_model
        elif attack_model_type == 'rf':
            if self._is_continuous:
                self.attack_model = RandomForestRegressor()
            else:
                self.attack_model = RandomForestClassifier()
        elif attack_model_type == 'gb':
            if self._is_continuous:
                self.attack_model = GradientBoostingRegressor()
            else:
                self.attack_model = GradientBoostingClassifier()
        elif attack_model_type == 'lr':
            if self._is_continuous:
                self.attack_model = LinearRegression()
            else:
                self.attack_model = LogisticRegression()
        elif attack_model_type == 'dt':
            if self._is_continuous:
                self.attack_model = DecisionTreeRegressor()
            else:
                self.attack_model = DecisionTreeClassifier()
        elif attack_model_type == 'knn':
            if self._is_continuous:
                self.attack_model = KNeighborsRegressor()
            else:
                self.attack_model = KNeighborsClassifier()
        elif attack_model_type == 'svm':
            if self._is_continuous:
                self.attack_model = SVR()
            else:
                self.attack_model = SVC(probability=True)
        elif attack_model_type != 'nn':
            raise ValueError('Illegal value for parameter `attack_model_type`.')
        self.prediction_normal_factor = prediction_normal_factor
        self.scale_range = scale_range
        self.is_regression = is_regression
        self._check_params()
        remove_attacked_feature(self.attack_feature, self._non_numerical_features)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if False:
            return 10
        '\n        Train the attack model.\n\n        :param x: Input to training process. Includes all features used to train the original model.\n        :param y: True labels of the features.\n        '
        if isinstance(self.attack_feature, int) and self.attack_feature >= x.shape[1]:
            raise ValueError('attack_feature must be a valid index to a feature in x')
        attacked_feature = x[:, self.attack_feature]
        y_ready = attacked_feature
        if not self._is_continuous:
            self._values = get_feature_values(attacked_feature, isinstance(self.attack_feature, int))
            nb_classes = len(self._values)
            if isinstance(self.attack_feature, int):
                y_one_hot = float_to_categorical(attacked_feature)
            else:
                y_one_hot = floats_to_one_hot(attacked_feature)
            y_ready = check_and_transform_label_format(y_one_hot, nb_classes=nb_classes, return_one_hot=True)
            if y_ready is None:
                raise ValueError('None value detected.')
            if self._attack_model_type in ('gb', 'lr', 'svm'):
                y_ready = np.argmax(y_ready, axis=1)
        if self.is_regression:
            if self.scale_range is not None:
                normalized_labels = minmax_scale(y, feature_range=self.scale_range)
            else:
                normalized_labels = y * self.prediction_normal_factor
            normalized_labels = normalized_labels.reshape(-1, 1)
        else:
            normalized_labels = check_and_transform_label_format(y, nb_classes=None, return_one_hot=True)
        x_train = np.delete(x, self.attack_feature, 1)
        if self._non_numerical_features and self._encoder is None:
            if isinstance(self.attack_feature, int):
                compare_index = self.attack_feature
                size = 1
            else:
                compare_index = self.attack_feature.start
                size = (self.attack_feature.stop - self.attack_feature.start) // self.attack_feature.step
            new_indexes = [f - size if f > compare_index else f for f in self._non_numerical_features]
            categorical_transformer = OrdinalEncoder()
            self._encoder = ColumnTransformer(transformers=[('cat', categorical_transformer, new_indexes)], remainder='passthrough')
            self._encoder.fit(x_train)
        if self._encoder is not None:
            x_train = self._encoder.transform(x_train)
        x_train = np.concatenate((x_train, normalized_labels), axis=1).astype(np.float32)
        if self._attack_model_type == 'nn':
            import torch
            from torch import nn
            from torch import optim
            from torch.utils.data import DataLoader
            from art.utils import to_cuda
            self.epochs = 100
            self.batch_size = 100
            self.learning_rate = 0.0001
            if self._is_continuous:

                class MembershipInferenceAttackModelRegression(nn.Module):
                    """
                    Implementation of a pytorch model for learning a membership inference attack.

                    The features used are probabilities/logits or losses for the attack training data along with
                    its true labels.
                    """

                    def __init__(self, num_features):
                        if False:
                            return 10
                        self.num_features = num_features
                        super().__init__()
                        self.features = nn.Sequential(nn.Linear(self.num_features, 100), nn.ReLU(), nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 1))

                    def forward(self, x):
                        if False:
                            for i in range(10):
                                print('nop')
                        'Forward the model.'
                        return self.features(x)
                self.attack_model = MembershipInferenceAttackModelRegression(x_train.shape[1])
                loss_fn: Any = nn.MSELoss()
            else:

                class MembershipInferenceAttackModel(nn.Module):
                    """
                    Implementation of a pytorch model for learning an attribute inference attack.

                    The features used are the remaining n-1 features of the attack training data along with
                    the model's predictions.
                    """

                    def __init__(self, num_features, num_classes):
                        if False:
                            return 10
                        self.num_classes = num_classes
                        self.num_features = num_features
                        super().__init__()
                        self.features = nn.Sequential(nn.Linear(self.num_features, 512), nn.ReLU(), nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, num_classes))
                        self.output = nn.Softmax()

                    def forward(self, x):
                        if False:
                            print('Hello World!')
                        'Forward the model.'
                        out = self.features(x)
                        return self.output(out)
                self.attack_model = MembershipInferenceAttackModel(x_train.shape[1], len(self._values))
                loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.attack_model.parameters(), lr=self.learning_rate)
            attack_train_set = self._get_attack_dataset(feature=x_train, label=y_ready)
            train_loader = DataLoader(attack_train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
            self.attack_model = to_cuda(self.attack_model)
            self.attack_model.train()
            for _ in range(self.epochs):
                for (input1, targets) in train_loader:
                    (input1, targets) = (to_cuda(input1), to_cuda(targets))
                    (_, targets) = (torch.autograd.Variable(input1), torch.autograd.Variable(targets))
                    optimizer.zero_grad()
                    outputs = self.attack_model(input1)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
        elif self.attack_model is not None:
            self.attack_model.fit(x_train, y_ready)

    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Infer the attacked feature.\n\n        :param x: Input to attack. Includes all features except the attacked feature.\n        :param y: True labels of the features.\n        :param values: Possible values for attacked feature. For a single column feature this should be a simple list\n                       containing all possible values, in increasing order (the smallest value in the 0 index and so\n                       on). For a multi-column feature (for example 1-hot encoded and then scaled), this should be a\n                       list of lists, where each internal list represents a column (in increasing order) and the values\n                       represent the possible values for that column (in increasing order).\n        :type values: list\n        :return: The inferred feature values.\n        '
        if y is None:
            raise ValueError('True labels are required')
        values = kwargs.get('values')
        if values is not None:
            self._values = values
        if self.is_regression:
            if self.scale_range is not None:
                normalized_labels = minmax_scale(y, feature_range=self.scale_range)
            else:
                normalized_labels = y * self.prediction_normal_factor
            normalized_labels = normalized_labels.reshape(-1, 1)
        else:
            normalized_labels = check_and_transform_label_format(y, nb_classes=None, return_one_hot=True)
        x_test = x
        if self._encoder is not None:
            x_test = self._encoder.transform(x)
        x_test = np.concatenate((x_test, normalized_labels), axis=1).astype(np.float32)
        if self._attack_model_type == 'nn':
            from torch.utils.data import DataLoader
            from art.utils import to_cuda, from_cuda
            self.attack_model.eval()
            predictions: np.ndarray = np.array([])
            test_set = self._get_attack_dataset(feature=x_test)
            test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
            for (input1, _) in test_loader:
                input1 = to_cuda(input1)
                outputs = self.attack_model(input1)
                predicted = from_cuda(outputs)
                if np.size(predictions) == 0:
                    predictions = predicted.detach().numpy()
                else:
                    predictions = np.vstack((predictions, predicted.detach().numpy()))
                if not self._is_continuous:
                    idx = np.argmax(predictions, axis=-1)
                    predictions = np.zeros(predictions.shape)
                    predictions[np.arange(predictions.shape[0]), idx] = 1
        elif self.attack_model is not None:
            predictions = self.attack_model.predict(x_test)
        if predictions is not None:
            predictions = predictions.astype(np.float32)
        if not self._is_continuous and self._values:
            if isinstance(self.attack_feature, int):
                if self._attack_model_type in ('gb', 'lr', 'svm'):
                    indexes = predictions
                else:
                    indexes = np.argmax(predictions, axis=1)
                predictions = np.array([self._values[int(index)] for index in indexes])
            else:
                if self._attack_model_type in ('gb', 'lr', 'svm'):
                    predictions = check_and_transform_label_format(predictions, nb_classes=len(self._values), return_one_hot=True)
                i = 0
                for column in predictions.T:
                    for index in range(len(self._values[i])):
                        np.place(column, [column == index], self._values[i][index])
                    i += 1
        return np.array(predictions)

    def _get_attack_dataset(self, feature, label=None):
        if False:
            print('Hello World!')
        from torch.utils.data.dataset import Dataset

        class AttackDataset(Dataset):
            """
            Implementation of a pytorch dataset for membership inference attack.

            The features are probabilities/logits or losses for the attack training data (`x_1`) along with
            its true labels (`x_2`). The labels (`y`) are a boolean representing whether this is a member.
            """

            def __init__(self, x, y=None):
                if False:
                    i = 10
                    return i + 15
                import torch
                self.x = torch.from_numpy(x.astype(np.float64)).type(torch.FloatTensor)
                if y is not None:
                    self.y = torch.from_numpy(y.astype(np.float32)).type(torch.FloatTensor)
                else:
                    self.y = torch.zeros(x.shape[0])

            def __len__(self):
                if False:
                    i = 10
                    return i + 15
                return len(self.x)

            def __getitem__(self, idx):
                if False:
                    print('Hello World!')
                if idx >= len(self.x):
                    raise IndexError('Invalid Index')
                return (self.x[idx], self.y[idx])
        return AttackDataset(x=feature, y=label)

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        super()._check_params()
        if not isinstance(self._is_continuous, bool):
            raise ValueError('is_continuous must be a boolean.')
        if self._attack_model_type not in ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm']:
            raise ValueError('Illegal value for parameter `attack_model_type`.')
        if self._non_numerical_features and (not isinstance(self._non_numerical_features, list) or not all((isinstance(item, int) for item in self._non_numerical_features))):
            raise ValueError('non_numerical_features must be a list of int.')
        if self._encoder is not None and (not isinstance(self._encoder, OrdinalEncoder) and (not isinstance(self._encoder, OneHotEncoder)) and (not isinstance(self._encoder, ColumnTransformer))):
            raise ValueError('encoder must be a OneHotEncoder, OrdinalEncoder or ColumnTransformer object.')