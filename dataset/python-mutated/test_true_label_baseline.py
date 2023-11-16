from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from art.attacks.inference.attribute_inference.true_label_baseline import AttributeInferenceBaselineTrueLabel
from art.utils import check_and_transform_label_format
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline(art_warning, get_iris_dataset, model_type):
    if False:
        return 10
    try:
        attack_feature = 2

        def transform_feature(x):
            if False:
                while True:
                    i = 10
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
        values = [0.0, 1.0, 2.0]
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type)
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y=y_train_iris, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y=y_test_iris, values=values)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.81, 'rf': 0.98, 'gb': 0.98, 'lr': 0.81, 'dt': 0.98, 'knn': 0.85, 'svm': 0.81}
        expected_test_acc = {'nn': 0.88, 'rf': 0.8, 'gb': 0.74, 'lr': 0.88, 'dt': 0.75, 'knn': 0.82, 'svm': 0.88}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_continuous(art_warning, get_iris_dataset, model_type):
    if False:
        while True:
            i = 10
    try:
        attack_feature = 2
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type, is_continuous=True)
        baseline_attack.fit(x_train_iris, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y=y_train_iris)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y=y_test_iris)
        assert np.count_nonzero(np.isclose(baseline_inferred_train, x_train_feature.reshape(1, -1), atol=0.4)) > baseline_inferred_train.shape[0] * 0.75
        assert np.count_nonzero(np.isclose(baseline_inferred_test, x_test_feature.reshape(1, -1), atol=0.4)) > baseline_inferred_test.shape[0] * 0.75
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_column(art_warning, get_iris_dataset, model_type):
    if False:
        print('Hello World!')
    try:
        attack_feature = 2

        def transform_feature(x):
            if False:
                return 10
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
        values = [0.0, 1.0, 2.0]
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        y_train_iris = y_train_iris[:, 0]
        y_test_iris = y_test_iris[:, 0]
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type)
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y=y_train_iris, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y=y_test_iris, values=values)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.81, 'rf': 0.98, 'gb': 0.98, 'lr': 0.81, 'dt': 0.98, 'knn': 0.87, 'svm': 0.81}
        expected_test_acc = {'nn': 0.88, 'rf': 0.8, 'gb': 0.82, 'lr': 0.88, 'dt': 0.75, 'knn': 0.84, 'svm': 0.88}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_no_values(art_warning, get_iris_dataset, model_type):
    if False:
        return 10
    try:
        attack_feature = 2

        def transform_feature(x):
            if False:
                print('Hello World!')
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type)
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y=y_train_iris)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y=y_test_iris)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.81, 'rf': 0.98, 'gb': 0.98, 'lr': 0.81, 'dt': 0.98, 'knn': 0.85, 'svm': 0.81}
        expected_test_acc = {'nn': 0.88, 'rf': 0.83, 'gb': 0.75, 'lr': 0.88, 'dt': 0.8, 'knn': 0.82, 'svm': 0.88}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
def test_true_label_baseline_slice(art_warning, get_iris_dataset):
    if False:
        while True:
            i = 10
    try:
        attack_feature = 2

        def transform_feature(x):
            if False:
                return 10
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
        values = [0.0, 1.0, 2.0]
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=slice(attack_feature, attack_feature + 1))
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y=y_train_iris, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y=y_test_iris, values=values)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        assert 0.8 <= baseline_train_acc
        assert 0.7 <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_regression(art_warning, get_diabetes_dataset, model_type):
    if False:
        while True:
            i = 10
    try:
        attack_feature = 1
        ((x_train, y_train), (x_test, y_test)) = get_diabetes_dataset
        x_train_for_attack = np.delete(x_train, attack_feature, 1)
        x_train_feature = x_train[:, attack_feature].copy().reshape(-1, 1)
        x_test_for_attack = np.delete(x_test, attack_feature, 1)
        x_test_feature = x_test[:, attack_feature].copy().reshape(-1, 1)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type, is_regression=True)
        baseline_attack.fit(x_train, y_train)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y=y_train)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y=y_test)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.45, 'rf': 0.99, 'gb': 0.97, 'lr': 0.68, 'dt': 0.99, 'knn': 0.69, 'svm': 0.54}
        expected_test_acc = {'nn': 0.45, 'rf': 0.6, 'gb': 0.65, 'lr': 0.68, 'dt': 0.54, 'knn': 0.45, 'svm': 0.47}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_encoder(art_warning, get_iris_dataset, model_type):
    if False:
        i = 10
        return i + 15
    try:
        attack_feature = 2

        def transform_attacked_feature(x):
            if False:
                while True:
                    i = 10
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            if False:
                while True:
                    i = 10
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = 'A'
            x[x == 1.0] = 'B'
            x[x == 0.0] = 'C'
        values = [0.0, 1.0, 2.0]
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_train_feature)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)
        other_feature = 1
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)
        x_train_for_attack_without_feature = np.delete(x_train_for_attack, other_feature, 1)
        x_train_for_attack = np.concatenate((x_train_for_attack_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train_for_attack = np.concatenate((x_train_for_attack, x_train_for_attack_without_feature[:, other_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_test_feature)
        x_test_without_feature = np.delete(x_test_for_attack, other_feature, 1)
        x_test_other_feature = x_test_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        x_test_for_attack = np.concatenate((x_test_without_feature[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, other_feature:]), axis=1)
        categorical_transformer = OrdinalEncoder()
        encoder = ColumnTransformer(transformers=[('cat', categorical_transformer, [other_feature])], remainder='passthrough')
        encoder.fit(x_train_for_attack)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type, encoder=encoder)
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y_train_iris, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y_test_iris, values=values)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.81, 'rf': 0.96, 'gb': 0.96, 'lr': 0.81, 'dt': 0.96, 'knn': 0.9, 'svm': 0.81}
        expected_test_acc = {'nn': 0.88, 'rf': 0.77, 'gb': 0.77, 'lr': 0.88, 'dt': 0.81, 'knn': 0.84, 'svm': 0.88}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_no_encoder(art_warning, get_iris_dataset, model_type):
    if False:
        print('Hello World!')
    try:
        attack_feature = 2

        def transform_attacked_feature(x):
            if False:
                for i in range(10):
                    print('nop')
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            if False:
                while True:
                    i = 10
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = 'A'
            x[x == 1.0] = 'B'
            x[x == 0.0] = 'C'
        values = [0.0, 1.0, 2.0]
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_train_feature)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)
        other_feature = 1
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)
        x_train_for_attack_without_feature = np.delete(x_train_for_attack, other_feature, 1)
        x_train_for_attack = np.concatenate((x_train_for_attack_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train_for_attack = np.concatenate((x_train_for_attack, x_train_for_attack_without_feature[:, other_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_test_feature)
        x_test_without_feature = np.delete(x_test_for_attack, other_feature, 1)
        x_test_other_feature = x_test_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        x_test_for_attack = np.concatenate((x_test_without_feature[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, other_feature:]), axis=1)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type, non_numerical_features=[other_feature])
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y_train_iris, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y_test_iris, values=values)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.81, 'rf': 0.96, 'gb': 0.96, 'lr': 0.81, 'dt': 0.96, 'knn': 0.9, 'svm': 0.81}
        expected_test_acc = {'nn': 0.88, 'rf': 0.81, 'gb': 0.77, 'lr': 0.88, 'dt': 0.82, 'knn': 0.84, 'svm': 0.88}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_no_encoder_after_feature(art_warning, get_iris_dataset, model_type):
    if False:
        while True:
            i = 10
    try:
        attack_feature = 2

        def transform_attacked_feature(x):
            if False:
                for i in range(10):
                    print('nop')
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            if False:
                while True:
                    i = 10
            x[x > 0.3] = 2.0
            x[(x > 0.2) & (x <= 0.3)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = 'A'
            x[x == 1.0] = 'B'
            x[x == 0.0] = 'C'
        values = [0.0, 1.0, 2.0]
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_train_feature)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)
        other_feature = 3
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)
        new_other_feature = other_feature - 1
        x_train_for_attack_without_feature = np.delete(x_train_for_attack, new_other_feature, 1)
        x_train_for_attack = np.concatenate((x_train_for_attack_without_feature[:, :new_other_feature], x_other_feature), axis=1)
        x_train_for_attack = np.concatenate((x_train_for_attack, x_train_for_attack_without_feature[:, new_other_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_test_feature)
        x_test_without_feature = np.delete(x_test_for_attack, new_other_feature, 1)
        x_test_other_feature = x_test_iris[:, new_other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        x_test_for_attack = np.concatenate((x_test_without_feature[:, :new_other_feature], x_test_other_feature), axis=1)
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, new_other_feature:]), axis=1)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type, non_numerical_features=[other_feature])
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y_train_iris, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y_test_iris, values=values)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.81, 'rf': 0.95, 'gb': 0.95, 'lr': 0.81, 'dt': 0.94, 'knn': 0.87, 'svm': 0.81}
        expected_test_acc = {'nn': 0.88, 'rf': 0.82, 'gb': 0.8, 'lr': 0.88, 'dt': 0.74, 'knn': 0.86, 'svm': 0.88}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_no_encoder_after_feature_slice(art_warning, get_iris_dataset, model_type):
    if False:
        while True:
            i = 10
    try:
        orig_attack_feature = 1
        new_attack_feature = slice(1, 4)

        def transform_attacked_feature(x):
            if False:
                print('Hello World!')
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            if False:
                print('Hello World!')
            x[x > 0.3] = 2.0
            x[(x > 0.2) & (x <= 0.3)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = 'A'
            x[x == 1.0] = 'B'
            x[x == 0.0] = 'C'
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, orig_attack_feature, 1)
        x_train_feature = x_train_iris[:, orig_attack_feature].copy()
        transform_attacked_feature(x_train_feature)
        x_train_feature = check_and_transform_label_format(x_train_feature, nb_classes=3, return_one_hot=True)
        x_train = np.concatenate((x_train_for_attack[:, :orig_attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, orig_attack_feature:]), axis=1)
        other_feature = 5
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)
        new_other_feature = other_feature - 3
        x_train_for_attack_without_feature = np.delete(x_train_for_attack, new_other_feature, 1)
        x_train_for_attack = np.concatenate((x_train_for_attack_without_feature[:, :new_other_feature], x_other_feature), axis=1)
        x_train_for_attack = np.concatenate((x_train_for_attack, x_train_for_attack_without_feature[:, new_other_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, orig_attack_feature, 1)
        x_test_feature = x_test_iris[:, orig_attack_feature].copy()
        transform_attacked_feature(x_test_feature)
        x_test_feature = check_and_transform_label_format(x_test_feature, nb_classes=3, return_one_hot=True)
        x_test_without_feature = np.delete(x_test_for_attack, new_other_feature, 1)
        x_test_other_feature = x_test_for_attack[:, new_other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        x_test_for_attack = np.concatenate((x_test_without_feature[:, :new_other_feature], x_test_other_feature), axis=1)
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, new_other_feature:]), axis=1)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=new_attack_feature, attack_model_type=model_type, non_numerical_features=[other_feature])
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = np.argmax(baseline_attack.infer(x_train_for_attack, y_train_iris), axis=1)
        baseline_inferred_test = np.argmax(baseline_attack.infer(x_test_for_attack, y_test_iris), axis=1)
        x_train_feature = np.argmax(x_train_feature, axis=1)
        x_test_feature = np.argmax(x_test_feature, axis=1)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.81, 'rf': 0.98, 'gb': 0.98, 'lr': 0.81, 'dt': 0.98, 'knn': 0.85, 'svm': 0.81}
        expected_test_acc = {'nn': 0.88, 'rf': 0.86, 'gb': 0.8, 'lr': 0.88, 'dt': 0.84, 'knn': 0.82, 'svm': 0.88}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
@pytest.mark.parametrize('model_type', ['nn', 'rf', 'gb', 'lr', 'dt', 'knn', 'svm'])
def test_true_label_baseline_no_encoder_remove_attack_feature(art_warning, get_iris_dataset, model_type):
    if False:
        while True:
            i = 10
    try:
        attack_feature = 2

        def transform_attacked_feature(x):
            if False:
                return 10
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            if False:
                print('Hello World!')
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = 'A'
            x[x == 1.0] = 'B'
            x[x == 0.0] = 'C'
        values = [0.0, 1.0, 2.0]
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_train_feature)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)
        other_feature = 1
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)
        x_train_for_attack_without_feature = np.delete(x_train_for_attack, other_feature, 1)
        x_train_for_attack = np.concatenate((x_train_for_attack_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train_for_attack = np.concatenate((x_train_for_attack, x_train_for_attack_without_feature[:, other_feature:]), axis=1)
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_test_feature)
        x_test_without_feature = np.delete(x_test_for_attack, other_feature, 1)
        x_test_other_feature = x_test_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        x_test_for_attack = np.concatenate((x_test_without_feature[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, other_feature:]), axis=1)
        baseline_attack = AttributeInferenceBaselineTrueLabel(attack_feature=attack_feature, attack_model_type=model_type, non_numerical_features=[other_feature, attack_feature])
        baseline_attack.fit(x_train, y_train_iris)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y_train_iris, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y_test_iris, values=values)
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(baseline_inferred_test)
        expected_train_acc = {'nn': 0.81, 'rf': 0.96, 'gb': 0.96, 'lr': 0.81, 'dt': 0.96, 'knn': 0.9, 'svm': 0.81}
        expected_test_acc = {'nn': 0.88, 'rf': 0.82, 'gb': 0.77, 'lr': 0.88, 'dt': 0.82, 'knn': 0.84, 'svm': 0.88}
        assert expected_train_acc[model_type] <= baseline_train_acc
        assert expected_test_acc[model_type] <= baseline_test_acc
    except ARTTestException as e:
        art_warning(e)

def test_check_params(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        with pytest.raises(ValueError):
            AttributeInferenceBaselineTrueLabel(attack_feature='a')
        with pytest.raises(ValueError):
            AttributeInferenceBaselineTrueLabel(attack_feature=-3)
        with pytest.raises(ValueError):
            AttributeInferenceBaselineTrueLabel(non_numerical_features=['a'])
        with pytest.raises(ValueError):
            AttributeInferenceBaselineTrueLabel(encoder='a')
        with pytest.raises(ValueError):
            AttributeInferenceBaselineTrueLabel(is_continuous='a')
    except ARTTestException as e:
        art_warning(e)