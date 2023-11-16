from typing import Sequence
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from darts import TimeSeries
from darts.ad.aggregators.and_aggregator import AndAggregator
from darts.ad.aggregators.ensemble_sklearn_aggregator import EnsembleSklearnAggregator
from darts.ad.aggregators.or_aggregator import OrAggregator
from darts.models import MovingAverageFilter
list_NonFittableAggregator = [OrAggregator(), AndAggregator()]
list_FittableAggregator = [EnsembleSklearnAggregator(model=GradientBoostingClassifier())]

class TestADAggregators:
    np.random.seed(42)
    np_train = np.random.normal(loc=10, scale=0.5, size=100)
    train = TimeSeries.from_values(np_train)
    np_real_anomalies = np.random.choice(a=[0, 1], size=100, p=[0.5, 0.5])
    real_anomalies = TimeSeries.from_times_and_values(train._time_index, np_real_anomalies)
    np_mts_train = np.random.normal(loc=[10, 5], scale=[0.5, 1], size=[100, 2])
    mts_train = TimeSeries.from_values(np_mts_train)
    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    mts_anomalies1 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)
    np_anomalies = np.random.choice(a=[0, 1], size=[100, 2], p=[0.5, 0.5])
    mts_anomalies2 = TimeSeries.from_times_and_values(train._time_index, np_anomalies)
    np_anomalies_w3 = np.random.choice(a=[0, 1], size=[100, 3], p=[0.5, 0.5])
    mts_anomalies3 = TimeSeries.from_times_and_values(train._time_index, np_anomalies_w3)
    np_probabilistic = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=[100, 2, 5])
    mts_probabilistic = TimeSeries.from_values(np_probabilistic)
    np_anomalies_1 = np.random.choice(a=[1], size=100, p=[1])
    onlyones = TimeSeries.from_times_and_values(train._time_index, np_anomalies_1)
    np_anomalies = np.random.choice(a=[1], size=[100, 2], p=[1])
    mts_onlyones = TimeSeries.from_times_and_values(train._time_index, np_anomalies)
    np_anomalies_0 = np.random.choice(a=[0], size=100, p=[1])
    onlyzero = TimeSeries.from_times_and_values(train._time_index, np_anomalies_0)
    series_1_and_0 = TimeSeries.from_values(np.dstack((np_anomalies_1, np_anomalies_0))[0], columns=['component 1', 'component 2'])
    np_real_anomalies_3w = [elem[0] if elem[2] == 1 else elem[1] for elem in np_anomalies_w3]
    real_anomalies_3w = TimeSeries.from_times_and_values(train._time_index, np_real_anomalies_3w)

    def test_DetectNonFittableAggregator(self):
        if False:
            for i in range(10):
                print('nop')
        aggregator = OrAggregator()
        assert isinstance(aggregator.predict(self.mts_anomalies1), TimeSeries)
        assert isinstance(aggregator.predict([self.mts_anomalies1]), Sequence)
        assert isinstance(aggregator.predict([self.mts_anomalies1, self.mts_anomalies2]), Sequence)

    def test_DetectFittableAggregator(self):
        if False:
            print('Hello World!')
        aggregator = EnsembleSklearnAggregator(model=GradientBoostingClassifier())
        aggregator.fit(self.real_anomalies, self.mts_anomalies1)
        assert isinstance(aggregator.predict(self.mts_anomalies1), TimeSeries)
        assert isinstance(aggregator.predict([self.mts_anomalies1]), Sequence)
        assert isinstance(aggregator.predict([self.mts_anomalies1, self.mts_anomalies2]), Sequence)

    def test_eval_accuracy(self):
        if False:
            return 10
        aggregator = AndAggregator()
        assert isinstance(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1), float)
        assert isinstance(aggregator.eval_accuracy([self.real_anomalies], [self.mts_anomalies1]), Sequence)
        assert isinstance(aggregator.eval_accuracy(self.real_anomalies, [self.mts_anomalies1]), Sequence)
        assert isinstance(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2]), Sequence)
        with pytest.raises(ValueError):
            aggregator.eval_accuracy(self.real_anomalies[:30], self.mts_anomalies1[40:])
        with pytest.raises(ValueError):
            aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies[:30]], [self.mts_anomalies1, self.mts_anomalies1[40:]])
        with pytest.raises(ValueError):
            aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, window=101)

    def test_NonFittableAggregator(self):
        if False:
            print('Hello World!')
        for aggregator in list_NonFittableAggregator:
            assert type(aggregator.__str__()) == str
            assert not aggregator.trainable
            with pytest.raises(ValueError):
                aggregator.predict([self.real_anomalies])
            with pytest.raises(ValueError):
                aggregator.predict(self.real_anomalies)
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, self.real_anomalies])
            with pytest.raises(ValueError):
                aggregator.predict(self.mts_train)
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, self.mts_train])
            with pytest.raises(ValueError):
                aggregator.predict(self.mts_probabilistic)
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, self.mts_probabilistic])
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, 'random'])
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, 1])
            assert len(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2])), len([self.mts_anomalies1, self.mts_anomalies2])

    def test_FittableAggregator(self):
        if False:
            i = 10
            return i + 15
        for aggregator in list_FittableAggregator:
            assert type(aggregator.__str__()) == str
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, self.mts_anomalies1])
            assert aggregator.trainable
            assert not aggregator._fit_called
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies3])
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, self.real_anomalies)
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, [self.real_anomalies])
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.real_anomalies])
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, self.mts_train)
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, [self.mts_train])
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_train])
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, self.mts_probabilistic)
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, [self.mts_probabilistic])
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_probabilistic])
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, 'random')
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, [self.mts_anomalies1, 'random'])
            with pytest.raises(ValueError):
                aggregator.fit(self.real_anomalies, [self.mts_anomalies1, 1])
            with pytest.raises(ValueError):
                aggregator.fit(self.mts_anomalies1, self.mts_anomalies1)
            with pytest.raises(ValueError):
                aggregator.fit([self.mts_anomalies1], [self.mts_anomalies1])
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.mts_anomalies1], [self.mts_anomalies1, self.mts_anomalies1])
            with pytest.raises(ValueError):
                aggregator.fit(self.train, self.mts_anomalies1)
            with pytest.raises(ValueError):
                aggregator.fit([self.train], self.mts_anomalies1)
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.train], [self.mts_anomalies1, self.mts_anomalies1])
            with pytest.raises(ValueError):
                aggregator.fit(self.mts_probabilistic, self.mts_anomalies1)
            with pytest.raises(ValueError):
                aggregator.fit([self.mts_probabilistic], self.mts_anomalies1)
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.mts_probabilistic], [self.mts_anomalies1, self.mts_anomalies1])
            with pytest.raises(ValueError):
                aggregator.fit('random', self.mts_anomalies1)
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, 'random'], [self.mts_anomalies1, self.mts_anomalies1])
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, 1], [self.mts_anomalies1, self.mts_anomalies1])
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.real_anomalies], self.mts_anomalies1)
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1])
            with pytest.raises(ValueError):
                aggregator.fit([self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies1])
            aggregator.fit(self.real_anomalies, self.mts_anomalies1)
            assert aggregator._fit_called
            with pytest.raises(ValueError):
                aggregator.predict(self.mts_anomalies3)
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies3])
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, self.mts_anomalies3])
            with pytest.raises(ValueError):
                aggregator.predict([self.real_anomalies])
            with pytest.raises(ValueError):
                aggregator.predict(self.real_anomalies)
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, self.real_anomalies])
            with pytest.raises(ValueError):
                aggregator.predict(self.mts_train)
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, self.mts_train])
            with pytest.raises(ValueError):
                aggregator.predict(self.mts_probabilistic)
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, self.mts_probabilistic])
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, 'random'])
            with pytest.raises(ValueError):
                aggregator.predict([self.mts_anomalies1, 1])
            assert len(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2])), len([self.mts_anomalies1, self.mts_anomalies2])

    def test_OrAggregator(self):
        if False:
            return 10
        aggregator = OrAggregator()
        assert abs(aggregator.eval_accuracy(self.onlyzero, self.series_1_and_0, metric='accuracy') - 0) < 1e-05
        assert abs(aggregator.eval_accuracy(self.onlyones, self.series_1_and_0, metric='accuracy') - 1) < 1e-05
        assert abs(aggregator.eval_accuracy(self.onlyones, self.mts_onlyones, metric='accuracy') - 1) < 1e-05
        assert abs(aggregator.eval_accuracy(self.onlyones, self.mts_onlyones, metric='recall') - 1) < 1e-05
        assert abs(aggregator.eval_accuracy(self.onlyones, self.mts_onlyones, metric='precision') - 1) < 1e-05
        assert aggregator.predict(self.mts_anomalies1).sum(axis=0).all_values().flatten()[0] == 67
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='accuracy') - 0.56) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='recall') - 0.72549) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='f1') - 0.62711) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='precision') - 0.55223) < 1e-05
        values = aggregator.predict([self.mts_anomalies1, self.mts_anomalies2])
        np.testing.assert_array_almost_equal([v.sum(axis=0).all_values().flatten()[0] for v in values], [67, 75], decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='accuracy')), np.array([0.56, 0.52]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='recall')), np.array([0.72549, 0.764706]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='f1')), np.array([0.627119, 0.619048]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='precision')), np.array([0.552239, 0.52]), decimal=1)

    def test_AndAggregator(self):
        if False:
            return 10
        aggregator = AndAggregator()
        assert abs(aggregator.eval_accuracy(self.onlyones, self.series_1_and_0, metric='accuracy') - 0) < 1e-05
        assert abs(aggregator.eval_accuracy(self.onlyzero, self.series_1_and_0, metric='accuracy') - 1) < 1e-05
        assert abs(aggregator.eval_accuracy(self.onlyones, self.mts_onlyones, metric='accuracy') - 1) < 1e-05
        assert abs(aggregator.eval_accuracy(self.onlyones, self.mts_onlyones, metric='recall') - 1) < 1e-05
        assert abs(aggregator.eval_accuracy(self.onlyones, self.mts_onlyones, metric='precision') - 1) < 1e-05
        assert aggregator.predict(self.mts_anomalies1).sum(axis=0).all_values().flatten()[0] == 27
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='accuracy') - 0.44) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='recall') - 0.21568) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='f1') - 0.28205) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='precision') - 0.4074) < 1e-05
        values = aggregator.predict([self.mts_anomalies1, self.mts_anomalies2])
        np.testing.assert_array_almost_equal([v.sum(axis=0).all_values().flatten()[0] for v in values], [27, 24], decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='accuracy')), np.array([0.44, 0.53]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='recall')), np.array([0.215686, 0.27451]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='f1')), np.array([0.282051, 0.373333]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='precision')), np.array([0.407407, 0.583333]), decimal=1)

    def test_EnsembleSklearn(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            EnsembleSklearnAggregator(model=MovingAverageFilter(window=10))
        aggregator = EnsembleSklearnAggregator(model=GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1))
        aggregator.fit(self.real_anomalies_3w, self.mts_anomalies3)
        assert abs(aggregator.eval_accuracy(self.real_anomalies_3w, self.mts_anomalies3, metric='accuracy') - 0.92) < 1e-05
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies_3w, self.real_anomalies_3w], [self.mts_anomalies3, self.mts_anomalies3], metric='accuracy')), np.array([0.92, 0.92]), decimal=1)
        aggregator = EnsembleSklearnAggregator(model=GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1))
        aggregator.fit(self.real_anomalies, self.mts_anomalies1)
        assert aggregator.predict(self.mts_anomalies1).sum(axis=0).all_values().flatten()[0] == 100
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='accuracy') - 0.51) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='recall') - 1.0) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='f1') - 0.67549) < 1e-05
        assert abs(aggregator.eval_accuracy(self.real_anomalies, self.mts_anomalies1, metric='precision') - 0.51) < 1e-05
        values = aggregator.predict([self.mts_anomalies1, self.mts_anomalies2])
        np.testing.assert_array_almost_equal([v.sum(axis=0).all_values().flatten()[0] for v in values], [100, 100.0], decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='accuracy')), np.array([0.51, 0.51]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='recall')), np.array([1, 1]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='f1')), np.array([0.675497, 0.675497]), decimal=1)
        np.testing.assert_array_almost_equal(np.array(aggregator.eval_accuracy([self.real_anomalies, self.real_anomalies], [self.mts_anomalies1, self.mts_anomalies2], metric='precision')), np.array([0.51, 0.51]), decimal=1)