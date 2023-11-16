from __future__ import division
from datetime import timedelta
from functools import partial
import blaze as bz
import itertools
from nose.tools import assert_true
from nose_parameterized import parameterized
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import pandas as pd
from toolz import merge
from zipline.pipeline import SimplePipelineEngine, Pipeline, CustomFactor
from zipline.pipeline.common import EVENT_DATE_FIELD_NAME, FISCAL_QUARTER_FIELD_NAME, FISCAL_YEAR_FIELD_NAME, SID_FIELD_NAME, TS_FIELD_NAME
from zipline.pipeline.data import DataSet
from zipline.pipeline.data import Column
from zipline.pipeline.domain import EquitySessionDomain
from zipline.pipeline.loaders.blaze.estimates import BlazeNextEstimatesLoader, BlazeNextSplitAdjustedEstimatesLoader, BlazePreviousEstimatesLoader, BlazePreviousSplitAdjustedEstimatesLoader
from zipline.pipeline.loaders.earnings_estimates import INVALID_NUM_QTRS_MESSAGE, NextEarningsEstimatesLoader, NextSplitAdjustedEarningsEstimatesLoader, normalize_quarters, PreviousEarningsEstimatesLoader, PreviousSplitAdjustedEarningsEstimatesLoader, split_normalized_quarters
from zipline.testing.fixtures import WithAdjustmentReader, WithTradingSessions, ZiplineTestCase
from zipline.testing.predicates import assert_equal, assert_raises_regex
from zipline.testing.predicates import assert_frame_equal
from zipline.utils.numpy_utils import datetime64ns_dtype
from zipline.utils.numpy_utils import float64_dtype

class Estimates(DataSet):
    event_date = Column(dtype=datetime64ns_dtype)
    fiscal_quarter = Column(dtype=float64_dtype)
    fiscal_year = Column(dtype=float64_dtype)
    estimate = Column(dtype=float64_dtype)

class MultipleColumnsEstimates(DataSet):
    event_date = Column(dtype=datetime64ns_dtype)
    fiscal_quarter = Column(dtype=float64_dtype)
    fiscal_year = Column(dtype=float64_dtype)
    estimate1 = Column(dtype=float64_dtype)
    estimate2 = Column(dtype=float64_dtype)

def QuartersEstimates(announcements_out):
    if False:
        while True:
            i = 10

    class QtrEstimates(Estimates):
        num_announcements = announcements_out
        name = Estimates
    return QtrEstimates

def MultipleColumnsQuartersEstimates(announcements_out):
    if False:
        while True:
            i = 10

    class QtrEstimates(MultipleColumnsEstimates):
        num_announcements = announcements_out
        name = Estimates
    return QtrEstimates

def QuartersEstimatesNoNumQuartersAttr(num_qtr):
    if False:
        print('Hello World!')

    class QtrEstimates(Estimates):
        name = Estimates
    return QtrEstimates

def create_expected_df_for_factor_compute(start_date, sids, tuples, end_date):
    if False:
        while True:
            i = 10
    '\n    Given a list of tuples of new data we get for each sid on each critical\n    date (when information changes), create a DataFrame that fills that\n    data through a date range ending at `end_date`.\n    '
    df = pd.DataFrame(tuples, columns=[SID_FIELD_NAME, 'estimate', 'knowledge_date'])
    df = df.pivot_table(columns=SID_FIELD_NAME, values='estimate', index='knowledge_date')
    df = df.reindex(pd.date_range(start_date, end_date))
    df.index = df.index.rename('knowledge_date')
    df['at_date'] = end_date.tz_localize('utc')
    df = df.set_index(['at_date', df.index.tz_localize('utc')]).ffill()
    new_sids = set(sids) - set(df.columns)
    df = df.reindex(columns=df.columns.union(new_sids))
    return df

class WithEstimates(WithTradingSessions, WithAdjustmentReader):
    """
    ZiplineTestCase mixin providing cls.loader and cls.events as class
    level fixtures.


    Methods
    -------
    make_loader(events, columns) -> PipelineLoader
        Method which returns the loader to be used throughout tests.

        events : pd.DataFrame
            The raw events to be used as input to the pipeline loader.
        columns : dict[str -> str]
            The dictionary mapping the names of BoundColumns to the
            associated column name in the events DataFrame.
    make_columns() -> dict[BoundColumn -> str]
       Method which returns a dictionary of BoundColumns mapped to the
       associated column names in the raw data.
    """
    START_DATE = pd.Timestamp('2014-12-28')
    END_DATE = pd.Timestamp('2015-02-04')

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('make_loader')

    @classmethod
    def make_events(cls):
        if False:
            while True:
                i = 10
        raise NotImplementedError('make_events')

    @classmethod
    def get_sids(cls):
        if False:
            print('Hello World!')
        return cls.events[SID_FIELD_NAME].unique()

    @classmethod
    def make_columns(cls):
        if False:
            return 10
        return {Estimates.event_date: 'event_date', Estimates.fiscal_quarter: 'fiscal_quarter', Estimates.fiscal_year: 'fiscal_year', Estimates.estimate: 'estimate'}

    def make_engine(self, loader=None):
        if False:
            while True:
                i = 10
        if loader is None:
            loader = self.loader
        return SimplePipelineEngine(lambda x: loader, self.asset_finder, default_domain=EquitySessionDomain(self.trading_days, self.ASSET_FINDER_COUNTRY_CODE))

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        cls.events = cls.make_events()
        cls.ASSET_FINDER_EQUITY_SIDS = cls.get_sids()
        cls.ASSET_FINDER_EQUITY_SYMBOLS = ['s' + str(n) for n in cls.ASSET_FINDER_EQUITY_SIDS]
        super(WithEstimates, cls).init_class_fixtures()
        cls.columns = cls.make_columns()
        cls.loader = cls.make_loader(cls.events, {column.name: val for (column, val) in cls.columns.items()})

class WithOneDayPipeline(WithEstimates):
    """
    ZiplineTestCase mixin providing cls.events as a class level fixture and
    defining a test for all inheritors to use.

    Attributes
    ----------
    events : pd.DataFrame
        A simple DataFrame with columns needed for estimates and a single sid
        and no other data.

    Tests
    ------
    test_wrong_num_announcements_passed()
        Tests that loading with an incorrect quarter number raises an error.
    test_no_num_announcements_attr()
        Tests that the loader throws an AssertionError if the dataset being
        loaded has no `num_announcements` attribute.
    """

    @classmethod
    def make_columns(cls):
        if False:
            for i in range(10):
                print('nop')
        return {MultipleColumnsEstimates.event_date: 'event_date', MultipleColumnsEstimates.fiscal_quarter: 'fiscal_quarter', MultipleColumnsEstimates.fiscal_year: 'fiscal_year', MultipleColumnsEstimates.estimate1: 'estimate1', MultipleColumnsEstimates.estimate2: 'estimate2'}

    @classmethod
    def make_events(cls):
        if False:
            while True:
                i = 10
        return pd.DataFrame({SID_FIELD_NAME: [0] * 2, TS_FIELD_NAME: [pd.Timestamp('2015-01-01'), pd.Timestamp('2015-01-06')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-10'), pd.Timestamp('2015-01-20')], 'estimate1': [1.0, 2.0], 'estimate2': [3.0, 4.0], FISCAL_QUARTER_FIELD_NAME: [1, 2], FISCAL_YEAR_FIELD_NAME: [2015, 2015]})

    @classmethod
    def make_expected_out(cls):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('make_expected_out')

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(WithOneDayPipeline, cls).init_class_fixtures()
        cls.sid0 = cls.asset_finder.retrieve_asset(0)
        cls.expected_out = cls.make_expected_out()

    def test_load_one_day(self):
        if False:
            print('Hello World!')
        dataset = MultipleColumnsQuartersEstimates(1)
        engine = self.make_engine()
        results = engine.run_pipeline(Pipeline({c.name: c.latest for c in dataset.columns}), start_date=pd.Timestamp('2015-01-15', tz='utc'), end_date=pd.Timestamp('2015-01-15', tz='utc'))
        assert_frame_equal(results, self.expected_out)

class PreviousWithOneDayPipeline(WithOneDayPipeline, ZiplineTestCase):
    """
    Tests that previous quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            print('Hello World!')
        return PreviousEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_out(cls):
        if False:
            return 10
        return pd.DataFrame({EVENT_DATE_FIELD_NAME: pd.Timestamp('2015-01-10'), 'estimate1': 1.0, 'estimate2': 3.0, FISCAL_QUARTER_FIELD_NAME: 1.0, FISCAL_YEAR_FIELD_NAME: 2015.0}, index=pd.MultiIndex.from_tuples(((pd.Timestamp('2015-01-15', tz='utc'), cls.sid0),)))

class NextWithOneDayPipeline(WithOneDayPipeline, ZiplineTestCase):
    """
    Tests that next quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            print('Hello World!')
        return NextEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_out(cls):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame({EVENT_DATE_FIELD_NAME: pd.Timestamp('2015-01-20'), 'estimate1': 2.0, 'estimate2': 4.0, FISCAL_QUARTER_FIELD_NAME: 2.0, FISCAL_YEAR_FIELD_NAME: 2015.0}, index=pd.MultiIndex.from_tuples(((pd.Timestamp('2015-01-15', tz='utc'), cls.sid0),)))
dummy_df = pd.DataFrame({SID_FIELD_NAME: 0}, columns=[SID_FIELD_NAME, TS_FIELD_NAME, EVENT_DATE_FIELD_NAME, FISCAL_QUARTER_FIELD_NAME, FISCAL_YEAR_FIELD_NAME, 'estimate'], index=[0])

class WithWrongLoaderDefinition(WithEstimates):
    """
    ZiplineTestCase mixin providing cls.events as a class level fixture and
    defining a test for all inheritors to use.

    Attributes
    ----------
    events : pd.DataFrame
        A simple DataFrame with columns needed for estimates and a single sid
        and no other data.

    Tests
    ------
    test_wrong_num_announcements_passed()
        Tests that loading with an incorrect quarter number raises an error.
    test_no_num_announcements_attr()
        Tests that the loader throws an AssertionError if the dataset being
        loaded has no `num_announcements` attribute.
    """

    @classmethod
    def make_events(cls):
        if False:
            return 10
        return dummy_df

    def test_wrong_num_announcements_passed(self):
        if False:
            for i in range(10):
                print('nop')
        bad_dataset1 = QuartersEstimates(-1)
        bad_dataset2 = QuartersEstimates(-2)
        good_dataset = QuartersEstimates(1)
        engine = self.make_engine()
        columns = {c.name + str(dataset.num_announcements): c.latest for dataset in (bad_dataset1, bad_dataset2, good_dataset) for c in dataset.columns}
        p = Pipeline(columns)
        with self.assertRaises(ValueError) as e:
            engine.run_pipeline(p, start_date=self.trading_days[0], end_date=self.trading_days[-1])
            assert_raises_regex(e, INVALID_NUM_QTRS_MESSAGE % '-1,-2')

    def test_no_num_announcements_attr(self):
        if False:
            while True:
                i = 10
        dataset = QuartersEstimatesNoNumQuartersAttr(1)
        engine = self.make_engine()
        p = Pipeline({c.name: c.latest for c in dataset.columns})
        with self.assertRaises(AttributeError):
            engine.run_pipeline(p, start_date=self.trading_days[0], end_date=self.trading_days[-1])

class PreviousWithWrongNumQuarters(WithWrongLoaderDefinition, ZiplineTestCase):
    """
    Tests that previous quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return PreviousEarningsEstimatesLoader(events, columns)

class NextWithWrongNumQuarters(WithWrongLoaderDefinition, ZiplineTestCase):
    """
    Tests that next quarter loader correctly breaks if an incorrect
    number of quarters is passed.
    """

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            i = 10
            return i + 15
        return NextEarningsEstimatesLoader(events, columns)
options = ['split_adjustments_loader', 'split_adjusted_column_names', 'split_adjusted_asof']

class WrongSplitsLoaderDefinition(WithEstimates, ZiplineTestCase):
    """
    Test class that tests that loaders break correctly when incorrectly
    instantiated.

    Tests
    -----
    test_extra_splits_columns_passed(SplitAdjustedEstimatesLoader)
        A test that checks that the loader correctly breaks when an
        unexpected column is passed in the list of split-adjusted columns.
    """

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(WithEstimates, cls).init_class_fixtures()

    @parameterized.expand(itertools.product((NextSplitAdjustedEarningsEstimatesLoader, PreviousSplitAdjustedEarningsEstimatesLoader)))
    def test_extra_splits_columns_passed(self, loader):
        if False:
            print('Hello World!')
        columns = {Estimates.event_date: 'event_date', Estimates.fiscal_quarter: 'fiscal_quarter', Estimates.fiscal_year: 'fiscal_year', Estimates.estimate: 'estimate'}
        with self.assertRaises(ValueError):
            loader(dummy_df, {column.name: val for (column, val) in columns.items()}, split_adjustments_loader=self.adjustment_reader, split_adjusted_column_names=['estimate', 'extra_col'], split_adjusted_asof=pd.Timestamp('2015-01-01'))

class WithEstimatesTimeZero(WithEstimates):
    """
    ZiplineTestCase mixin providing cls.events as a class level fixture and
    defining a test for all inheritors to use.

    Attributes
    ----------
    cls.events : pd.DataFrame
        Generated dynamically in order to test inter-leavings of estimates and
        event dates for multiple quarters to make sure that we select the
        right immediate 'next' or 'previous' quarter relative to each date -
        i.e., the right 'time zero' on the timeline. We care about selecting
        the right 'time zero' because we use that to calculate which quarter's
        data needs to be returned for each day.

    Methods
    -------
    get_expected_estimate(q1_knowledge,
                          q2_knowledge,
                          comparable_date) -> pd.DataFrame
        Retrieves the expected estimate given the latest knowledge about each
        quarter and the date on which the estimate is being requested. If
        there is no expected estimate, returns an empty DataFrame.

    Tests
    ------
    test_estimates()
        Tests that we get the right 'time zero' value on each day for each
        sid and for each column.
    """
    END_DATE = pd.Timestamp('2015-01-28')
    q1_knowledge_dates = [pd.Timestamp('2015-01-01'), pd.Timestamp('2015-01-04'), pd.Timestamp('2015-01-07'), pd.Timestamp('2015-01-11')]
    q2_knowledge_dates = [pd.Timestamp('2015-01-14'), pd.Timestamp('2015-01-17'), pd.Timestamp('2015-01-20'), pd.Timestamp('2015-01-23')]
    q1_release_dates = [pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-14')]
    q2_release_dates = [pd.Timestamp('2015-01-25'), pd.Timestamp('2015-01-26')]

    @classmethod
    def make_events(cls):
        if False:
            return 10
        '\n        In order to determine which estimate we care about for a particular\n        sid, we need to look at all estimates that we have for that sid and\n        their associated event dates.\n\n        We define q1 < q2, and thus event1 < event2 since event1 occurs\n        during q1 and event2 occurs during q2 and we assume that there can\n        only be 1 event per quarter. We assume that there can be multiple\n        estimates per quarter leading up to the event. We assume that estimates\n        will not surpass the relevant event date. We will look at 2 estimates\n        for an event before the event occurs, since that is the simplest\n        scenario that covers the interesting edge cases:\n            - estimate values changing\n            - a release date changing\n            - estimates for different quarters interleaving\n\n        Thus, we generate all possible inter-leavings of 2 estimates per\n        quarter-event where estimate1 < estimate2 and all estimates are < the\n        relevant event and assign each of these inter-leavings to a\n        different sid.\n        '
        sid_estimates = []
        sid_releases = []
        it = enumerate(itertools.permutations(cls.q1_knowledge_dates + cls.q2_knowledge_dates, 4))
        for (sid, (q1e1, q1e2, q2e1, q2e2)) in it:
            if q1e1 < q1e2 and q2e1 < q2e2 and (q1e1 < cls.q1_release_dates[0]) and (q1e2 < cls.q1_release_dates[0]):
                sid_estimates.append(cls.create_estimates_df(q1e1, q1e2, q2e1, q2e2, sid))
                sid_releases.append(cls.create_releases_df(sid))
        return pd.concat(sid_estimates + sid_releases).reset_index(drop=True)

    @classmethod
    def get_sids(cls):
        if False:
            while True:
                i = 10
        sids = cls.events[SID_FIELD_NAME].unique()
        return list(sids) + [max(sids) + 1]

    @classmethod
    def create_releases_df(cls, sid):
        if False:
            return 10
        return pd.DataFrame({TS_FIELD_NAME: [pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-26')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-26')], 'estimate': [0.5, 0.8], FISCAL_QUARTER_FIELD_NAME: [1.0, 2.0], FISCAL_YEAR_FIELD_NAME: [2015.0, 2015.0], SID_FIELD_NAME: sid})

    @classmethod
    def create_estimates_df(cls, q1e1, q1e2, q2e1, q2e2, sid):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame({EVENT_DATE_FIELD_NAME: cls.q1_release_dates + cls.q2_release_dates, 'estimate': [0.1, 0.2, 0.3, 0.4], FISCAL_QUARTER_FIELD_NAME: [1.0, 1.0, 2.0, 2.0], FISCAL_YEAR_FIELD_NAME: [2015.0, 2015.0, 2015.0, 2015.0], TS_FIELD_NAME: [q1e1, q1e2, q2e1, q2e2], SID_FIELD_NAME: sid})

    def get_expected_estimate(self, q1_knowledge, q2_knowledge, comparable_date):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame()

    def test_estimates(self):
        if False:
            while True:
                i = 10
        dataset = QuartersEstimates(1)
        engine = self.make_engine()
        results = engine.run_pipeline(Pipeline({c.name: c.latest for c in dataset.columns}), start_date=self.trading_days[1], end_date=self.trading_days[-2])
        for sid in self.ASSET_FINDER_EQUITY_SIDS:
            sid_estimates = results.xs(sid, level=1)
            if sid == max(self.ASSET_FINDER_EQUITY_SIDS):
                assert_true(sid_estimates.isnull().all().all())
            else:
                ts_sorted_estimates = self.events[self.events[SID_FIELD_NAME] == sid].sort_values(TS_FIELD_NAME)
                q1_knowledge = ts_sorted_estimates[ts_sorted_estimates[FISCAL_QUARTER_FIELD_NAME] == 1]
                q2_knowledge = ts_sorted_estimates[ts_sorted_estimates[FISCAL_QUARTER_FIELD_NAME] == 2]
                all_expected = pd.concat([self.get_expected_estimate(q1_knowledge[q1_knowledge[TS_FIELD_NAME] <= date.tz_localize(None)], q2_knowledge[q2_knowledge[TS_FIELD_NAME] <= date.tz_localize(None)], date.tz_localize(None)).set_index([[date]]) for date in sid_estimates.index], axis=0)
                assert_equal(all_expected[sid_estimates.columns], sid_estimates)

class NextEstimate(WithEstimatesTimeZero, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return NextEarningsEstimatesLoader(events, columns)

    def get_expected_estimate(self, q1_knowledge, q2_knowledge, comparable_date):
        if False:
            while True:
                i = 10
        if not q1_knowledge.empty and q1_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] >= comparable_date:
            return q1_knowledge.iloc[-1:]
        elif not q2_knowledge.empty and q2_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] >= comparable_date:
            return q2_knowledge.iloc[-1:]
        return pd.DataFrame(columns=q1_knowledge.columns, index=[comparable_date])

class BlazeNextEstimateLoaderTestCase(NextEstimate):
    """
    Run the same tests as EventsLoaderTestCase, but using a BlazeEventsLoader.
    """

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            for i in range(10):
                print('nop')
        return BlazeNextEstimatesLoader(bz.data(events), columns)

class PreviousEstimate(WithEstimatesTimeZero, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return PreviousEarningsEstimatesLoader(events, columns)

    def get_expected_estimate(self, q1_knowledge, q2_knowledge, comparable_date):
        if False:
            return 10
        if not q2_knowledge.empty and q2_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] <= comparable_date:
            return q2_knowledge.iloc[-1:]
        elif not q1_knowledge.empty and q1_knowledge[EVENT_DATE_FIELD_NAME].iloc[-1] <= comparable_date:
            return q1_knowledge.iloc[-1:]
        return pd.DataFrame(columns=q1_knowledge.columns, index=[comparable_date])

class BlazePreviousEstimateLoaderTestCase(PreviousEstimate):
    """
    Run the same tests as EventsLoaderTestCase, but using a BlazeEventsLoader.
    """

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return BlazePreviousEstimatesLoader(bz.data(events), columns)

class WithEstimateMultipleQuarters(WithEstimates):
    """
    ZiplineTestCase mixin providing cls.events, cls.make_expected_out as
    class-level fixtures and self.test_multiple_qtrs_requested as a test.

    Attributes
    ----------
    events : pd.DataFrame
        Simple DataFrame with estimates for 2 quarters for a single sid.

    Methods
    -------
    make_expected_out() --> pd.DataFrame
        Returns the DataFrame that is expected as a result of running a
        Pipeline where estimates are requested for multiple quarters out.
    fill_expected_out(expected)
        Fills the expected DataFrame with data.

    Tests
    ------
    test_multiple_qtrs_requested()
        Runs a Pipeline that calculate which estimates for multiple quarters
        out and checks that the returned columns contain data for the correct
        number of quarters out.
    """

    @classmethod
    def make_events(cls):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame({SID_FIELD_NAME: [0] * 2, TS_FIELD_NAME: [pd.Timestamp('2015-01-01'), pd.Timestamp('2015-01-06')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-10'), pd.Timestamp('2015-01-20')], 'estimate': [1.0, 2.0], FISCAL_QUARTER_FIELD_NAME: [1, 2], FISCAL_YEAR_FIELD_NAME: [2015, 2015]})

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        super(WithEstimateMultipleQuarters, cls).init_class_fixtures()
        cls.expected_out = cls.make_expected_out()

    @classmethod
    def make_expected_out(cls):
        if False:
            i = 10
            return i + 15
        expected = pd.DataFrame(columns=[cls.columns[col] + '1' for col in cls.columns] + [cls.columns[col] + '2' for col in cls.columns], index=cls.trading_days)
        for ((col, raw_name), suffix) in itertools.product(cls.columns.items(), ('1', '2')):
            expected_name = raw_name + suffix
            if col.dtype == datetime64ns_dtype:
                expected[expected_name] = pd.to_datetime(expected[expected_name])
            else:
                expected[expected_name] = expected[expected_name].astype(col.dtype)
        cls.fill_expected_out(expected)
        return expected.reindex(cls.trading_days)

    def test_multiple_qtrs_requested(self):
        if False:
            return 10
        dataset1 = QuartersEstimates(1)
        dataset2 = QuartersEstimates(2)
        engine = self.make_engine()
        results = engine.run_pipeline(Pipeline(merge([{c.name + '1': c.latest for c in dataset1.columns}, {c.name + '2': c.latest for c in dataset2.columns}])), start_date=self.trading_days[0], end_date=self.trading_days[-1])
        q1_columns = [col.name + '1' for col in self.columns]
        q2_columns = [col.name + '2' for col in self.columns]
        assert_equal(sorted(np.array(q1_columns + q2_columns)), sorted(results.columns.values))
        assert_equal(self.expected_out.sort_index(axis=1), results.xs(0, level=1).sort_index(axis=1))

class NextEstimateMultipleQuarters(WithEstimateMultipleQuarters, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            while True:
                i = 10
        return NextEarningsEstimatesLoader(events, columns)

    @classmethod
    def fill_expected_out(cls, expected):
        if False:
            return 10
        for raw_name in cls.columns.values():
            expected.loc[pd.Timestamp('2015-01-01'):pd.Timestamp('2015-01-11'), raw_name + '1'] = cls.events[raw_name].iloc[0]
            expected.loc[pd.Timestamp('2015-01-11'):pd.Timestamp('2015-01-20'), raw_name + '1'] = cls.events[raw_name].iloc[1]
        for col_name in ['estimate', 'event_date']:
            expected.loc[pd.Timestamp('2015-01-06'):pd.Timestamp('2015-01-10'), col_name + '2'] = cls.events[col_name].iloc[1]
        expected.loc[pd.Timestamp('2015-01-01'):pd.Timestamp('2015-01-09'), FISCAL_QUARTER_FIELD_NAME + '2'] = 2
        expected.loc[pd.Timestamp('2015-01-12'):pd.Timestamp('2015-01-20'), FISCAL_QUARTER_FIELD_NAME + '2'] = 3
        expected.loc[pd.Timestamp('2015-01-01'):pd.Timestamp('2015-01-20'), FISCAL_YEAR_FIELD_NAME + '2'] = 2015
        return expected

class BlazeNextEstimateMultipleQuarters(NextEstimateMultipleQuarters):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            for i in range(10):
                print('nop')
        return BlazeNextEstimatesLoader(bz.data(events), columns)

class PreviousEstimateMultipleQuarters(WithEstimateMultipleQuarters, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return PreviousEarningsEstimatesLoader(events, columns)

    @classmethod
    def fill_expected_out(cls, expected):
        if False:
            for i in range(10):
                print('nop')
        for raw_name in cls.columns.values():
            expected[raw_name + '1'].loc[pd.Timestamp('2015-01-12'):pd.Timestamp('2015-01-19')] = cls.events[raw_name].iloc[0]
            expected[raw_name + '1'].loc[pd.Timestamp('2015-01-20'):] = cls.events[raw_name].iloc[1]
        for col_name in ['estimate', 'event_date']:
            expected[col_name + '2'].loc[pd.Timestamp('2015-01-20'):] = cls.events[col_name].iloc[0]
        expected[FISCAL_QUARTER_FIELD_NAME + '2'].loc[pd.Timestamp('2015-01-12'):pd.Timestamp('2015-01-20')] = 4
        expected[FISCAL_YEAR_FIELD_NAME + '2'].loc[pd.Timestamp('2015-01-12'):pd.Timestamp('2015-01-20')] = 2014
        expected[FISCAL_QUARTER_FIELD_NAME + '2'].loc[pd.Timestamp('2015-01-20'):] = 1
        expected[FISCAL_YEAR_FIELD_NAME + '2'].loc[pd.Timestamp('2015-01-20'):] = 2015
        return expected

class BlazePreviousEstimateMultipleQuarters(PreviousEstimateMultipleQuarters):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            while True:
                i = 10
        return BlazePreviousEstimatesLoader(bz.data(events), columns)

class WithVaryingNumEstimates(WithEstimates):
    """
    ZiplineTestCase mixin providing fixtures and a test to ensure that we
    have the correct overwrites when the event date changes. We want to make
    sure that if we have a quarter with an event date that gets pushed back,
    we don't start overwriting for the next quarter early. Likewise,
    if we have a quarter with an event date that gets pushed forward, we want
    to make sure that we start applying adjustments at the appropriate, earlier
    date, rather than the later date.

    Methods
    -------
    assert_compute()
        Defines how to determine that results computed for the `SomeFactor`
        factor are correct.

    Tests
    -----
    test_windows_with_varying_num_estimates()
        Tests that we create the correct overwrites from 2015-01-13 to
        2015-01-14 regardless of how event dates were updated for each
        quarter for each sid.
    """

    @classmethod
    def make_events(cls):
        if False:
            return 10
        return pd.DataFrame({SID_FIELD_NAME: [0] * 3 + [1] * 3, TS_FIELD_NAME: [pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-12'), pd.Timestamp('2015-01-13')] * 2, EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-12'), pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-20'), pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-12'), pd.Timestamp('2015-01-20')], 'estimate': [11.0, 12.0, 21.0] * 2, FISCAL_QUARTER_FIELD_NAME: [1, 1, 2] * 2, FISCAL_YEAR_FIELD_NAME: [2015] * 6})

    @classmethod
    def assert_compute(cls, estimate, today):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('assert_compute')

    def test_windows_with_varying_num_estimates(self):
        if False:
            print('Hello World!')
        dataset = QuartersEstimates(1)
        assert_compute = self.assert_compute

        class SomeFactor(CustomFactor):
            inputs = [dataset.estimate]
            window_length = 3

            def compute(self, today, assets, out, estimate):
                if False:
                    return 10
                assert_compute(estimate, today)
        engine = self.make_engine()
        engine.run_pipeline(Pipeline({'est': SomeFactor()}), start_date=pd.Timestamp('2015-01-13', tz='utc'), end_date=pd.Timestamp('2015-01-14', tz='utc'))

class PreviousVaryingNumEstimates(WithVaryingNumEstimates, ZiplineTestCase):

    def assert_compute(self, estimate, today):
        if False:
            for i in range(10):
                print('nop')
        if today == pd.Timestamp('2015-01-13', tz='utc'):
            assert_array_equal(estimate[:, 0], np.array([np.NaN, np.NaN, 12]))
            assert_array_equal(estimate[:, 1], np.array([np.NaN, 12, 12]))
        else:
            assert_array_equal(estimate[:, 0], np.array([np.NaN, 12, 12]))
            assert_array_equal(estimate[:, 1], np.array([12, 12, 12]))

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            while True:
                i = 10
        return PreviousEarningsEstimatesLoader(events, columns)

class BlazePreviousVaryingNumEstimates(PreviousVaryingNumEstimates):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            for i in range(10):
                print('nop')
        return BlazePreviousEstimatesLoader(bz.data(events), columns)

class NextVaryingNumEstimates(WithVaryingNumEstimates, ZiplineTestCase):

    def assert_compute(self, estimate, today):
        if False:
            while True:
                i = 10
        if today == pd.Timestamp('2015-01-13', tz='utc'):
            assert_array_equal(estimate[:, 0], np.array([11, 12, 12]))
            assert_array_equal(estimate[:, 1], np.array([np.NaN, np.NaN, 21]))
        else:
            assert_array_equal(estimate[:, 0], np.array([np.NaN, 21, 21]))
            assert_array_equal(estimate[:, 1], np.array([np.NaN, 21, 21]))

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            print('Hello World!')
        return NextEarningsEstimatesLoader(events, columns)

class BlazeNextVaryingNumEstimates(NextVaryingNumEstimates):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            print('Hello World!')
        return BlazeNextEstimatesLoader(bz.data(events), columns)

class WithEstimateWindows(WithEstimates):
    """
    ZiplineTestCase mixin providing fixures and a test to test running a
    Pipeline with an estimates loader over differently-sized windows.

    Attributes
    ----------
    events : pd.DataFrame
        DataFrame with estimates for 2 quarters for 2 sids.
    window_test_start_date : pd.Timestamp
        The date from which the window should start.
    timelines : dict[int -> pd.DataFrame]
        A dictionary mapping to the number of quarters out to
        snapshots of how the data should look on each date in the date range.

    Methods
    -------
    make_expected_timelines() -> dict[int -> pd.DataFrame]
        Creates a dictionary of expected data. See `timelines`, above.

    Tests
    -----
    test_estimate_windows_at_quarter_boundaries()
        Tests that we overwrite values with the correct quarter's estimate at
        the correct dates when we have a factor that asks for a window of data.
    """
    END_DATE = pd.Timestamp('2015-02-10')
    window_test_start_date = pd.Timestamp('2015-01-05')
    critical_dates = [pd.Timestamp('2015-01-09', tz='utc'), pd.Timestamp('2015-01-15', tz='utc'), pd.Timestamp('2015-01-20', tz='utc'), pd.Timestamp('2015-01-26', tz='utc'), pd.Timestamp('2015-02-05', tz='utc'), pd.Timestamp('2015-02-10', tz='utc')]
    window_test_cases = list(itertools.product(critical_dates, (1, 2)))

    @classmethod
    def make_events(cls):
        if False:
            print('Hello World!')
        sid_0_timeline = pd.DataFrame({TS_FIELD_NAME: [cls.window_test_start_date, pd.Timestamp('2015-01-20'), pd.Timestamp('2015-01-12'), pd.Timestamp('2015-02-10'), pd.Timestamp('2015-01-18')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-20'), pd.Timestamp('2015-01-20'), pd.Timestamp('2015-02-10'), pd.Timestamp('2015-02-10'), pd.Timestamp('2015-04-01')], 'estimate': [100.0, 101.0] + [200.0, 201.0] + [400], FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [2] * 2 + [4], FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 0})
        sid_10_timeline = pd.DataFrame({TS_FIELD_NAME: [pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-12'), pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-15')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-22'), pd.Timestamp('2015-01-22'), pd.Timestamp('2015-02-05'), pd.Timestamp('2015-02-05')], 'estimate': [110.0, 111.0] + [310.0, 311.0], FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [3] * 2, FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 10})
        sid_20_timeline = pd.DataFrame({TS_FIELD_NAME: [cls.window_test_start_date, pd.Timestamp('2015-01-07'), cls.window_test_start_date, pd.Timestamp('2015-01-17')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-20'), pd.Timestamp('2015-01-20'), pd.Timestamp('2015-02-10'), pd.Timestamp('2015-02-10')], 'estimate': [120.0, 121.0] + [220.0, 221.0], FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [2] * 2, FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 20})
        concatted = pd.concat([sid_0_timeline, sid_10_timeline, sid_20_timeline]).reset_index()
        np.random.seed(0)
        return concatted.reindex(np.random.permutation(concatted.index))

    @classmethod
    def get_sids(cls):
        if False:
            return 10
        sids = sorted(cls.events[SID_FIELD_NAME].unique())
        return [sid for i in range(len(sids) - 1) for sid in range(sids[i], sids[i + 1])] + [sids[-1]]

    @classmethod
    def make_expected_timelines(cls):
        if False:
            for i in range(10):
                print('nop')
        return {}

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(WithEstimateWindows, cls).init_class_fixtures()
        cls.create_expected_df_for_factor_compute = partial(create_expected_df_for_factor_compute, cls.window_test_start_date, cls.get_sids())
        cls.timelines = cls.make_expected_timelines()

    @parameterized.expand(window_test_cases)
    def test_estimate_windows_at_quarter_boundaries(self, start_date, num_announcements_out):
        if False:
            while True:
                i = 10
        dataset = QuartersEstimates(num_announcements_out)
        trading_days = self.trading_days
        timelines = self.timelines
        window_len = self.trading_days.get_loc(start_date) - self.trading_days.get_loc(self.window_test_start_date) + 1

        class SomeFactor(CustomFactor):
            inputs = [dataset.estimate]
            window_length = window_len

            def compute(self, today, assets, out, estimate):
                if False:
                    return 10
                today_idx = trading_days.get_loc(today)
                today_timeline = timelines[num_announcements_out].loc[today].reindex(trading_days[:today_idx + 1]).values
                timeline_start_idx = len(today_timeline) - window_len
                assert_almost_equal(estimate, today_timeline[timeline_start_idx:])
        engine = self.make_engine()
        engine.run_pipeline(Pipeline({'est': SomeFactor()}), start_date=start_date, end_date=pd.Timestamp('2015-02-10', tz='utc'))

class PreviousEstimateWindows(WithEstimateWindows, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            while True:
                i = 10
        return PreviousEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_timelines(cls):
        if False:
            while True:
                i = 10
        oneq_previous = pd.concat([pd.concat([cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-09', '2015-01-19')]), cls.create_expected_df_for_factor_compute([(0, 101, pd.Timestamp('2015-01-20')), (10, np.NaN, cls.window_test_start_date), (20, 121, pd.Timestamp('2015-01-20'))], pd.Timestamp('2015-01-20')), cls.create_expected_df_for_factor_compute([(0, 101, pd.Timestamp('2015-01-20')), (10, np.NaN, cls.window_test_start_date), (20, 121, pd.Timestamp('2015-01-20'))], pd.Timestamp('2015-01-21')), pd.concat([cls.create_expected_df_for_factor_compute([(0, 101, pd.Timestamp('2015-01-20')), (10, 111, pd.Timestamp('2015-01-22')), (20, 121, pd.Timestamp('2015-01-20'))], end_date) for end_date in pd.date_range('2015-01-22', '2015-02-04')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 101, pd.Timestamp('2015-01-20')), (10, 311, pd.Timestamp('2015-02-05')), (20, 121, pd.Timestamp('2015-01-20'))], end_date) for end_date in pd.date_range('2015-02-05', '2015-02-09')]), cls.create_expected_df_for_factor_compute([(0, 201, pd.Timestamp('2015-02-10')), (10, 311, pd.Timestamp('2015-02-05')), (20, 221, pd.Timestamp('2015-02-10'))], pd.Timestamp('2015-02-10'))])
        twoq_previous = pd.concat([cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-09', '2015-02-09')] + [cls.create_expected_df_for_factor_compute([(0, 101, pd.Timestamp('2015-02-10')), (10, np.NaN, pd.Timestamp('2015-02-05')), (20, 121, pd.Timestamp('2015-02-10'))], pd.Timestamp('2015-02-10'))])
        return {1: oneq_previous, 2: twoq_previous}

class BlazePreviousEstimateWindows(PreviousEstimateWindows):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            i = 10
            return i + 15
        return BlazePreviousEstimatesLoader(bz.data(events), columns)

class NextEstimateWindows(WithEstimateWindows, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return NextEarningsEstimatesLoader(events, columns)

    @classmethod
    def make_expected_timelines(cls):
        if False:
            for i in range(10):
                print('nop')
        oneq_next = pd.concat([cls.create_expected_df_for_factor_compute([(0, 100, cls.window_test_start_date), (10, 110, pd.Timestamp('2015-01-09')), (20, 120, cls.window_test_start_date), (20, 121, pd.Timestamp('2015-01-07'))], pd.Timestamp('2015-01-09')), pd.concat([cls.create_expected_df_for_factor_compute([(0, 100, cls.window_test_start_date), (10, 110, pd.Timestamp('2015-01-09')), (10, 111, pd.Timestamp('2015-01-12')), (20, 120, cls.window_test_start_date), (20, 121, pd.Timestamp('2015-01-07'))], end_date) for end_date in pd.date_range('2015-01-12', '2015-01-19')]), cls.create_expected_df_for_factor_compute([(0, 100, cls.window_test_start_date), (0, 101, pd.Timestamp('2015-01-20')), (10, 110, pd.Timestamp('2015-01-09')), (10, 111, pd.Timestamp('2015-01-12')), (20, 120, cls.window_test_start_date), (20, 121, pd.Timestamp('2015-01-07'))], pd.Timestamp('2015-01-20')), pd.concat([cls.create_expected_df_for_factor_compute([(0, 200, pd.Timestamp('2015-01-12')), (10, 110, pd.Timestamp('2015-01-09')), (10, 111, pd.Timestamp('2015-01-12')), (20, 220, cls.window_test_start_date), (20, 221, pd.Timestamp('2015-01-17'))], end_date) for end_date in pd.date_range('2015-01-21', '2015-01-22')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 200, pd.Timestamp('2015-01-12')), (10, 310, pd.Timestamp('2015-01-09')), (10, 311, pd.Timestamp('2015-01-15')), (20, 220, cls.window_test_start_date), (20, 221, pd.Timestamp('2015-01-17'))], end_date) for end_date in pd.date_range('2015-01-23', '2015-02-05')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 200, pd.Timestamp('2015-01-12')), (10, np.NaN, cls.window_test_start_date), (20, 220, cls.window_test_start_date), (20, 221, pd.Timestamp('2015-01-17'))], end_date) for end_date in pd.date_range('2015-02-06', '2015-02-09')]), cls.create_expected_df_for_factor_compute([(0, 200, pd.Timestamp('2015-01-12')), (0, 201, pd.Timestamp('2015-02-10')), (10, np.NaN, cls.window_test_start_date), (20, 220, cls.window_test_start_date), (20, 221, pd.Timestamp('2015-01-17'))], pd.Timestamp('2015-02-10'))])
        twoq_next = pd.concat([cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, 220, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-09', '2015-01-11')] + [cls.create_expected_df_for_factor_compute([(0, 200, pd.Timestamp('2015-01-12')), (10, np.NaN, cls.window_test_start_date), (20, 220, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-12', '2015-01-16')] + [cls.create_expected_df_for_factor_compute([(0, 200, pd.Timestamp('2015-01-12')), (10, np.NaN, cls.window_test_start_date), (20, 220, cls.window_test_start_date), (20, 221, pd.Timestamp('2015-01-17'))], pd.Timestamp('2015-01-20'))] + [cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-21', '2015-02-10')])
        return {1: oneq_next, 2: twoq_next}

class BlazeNextEstimateWindows(NextEstimateWindows):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return BlazeNextEstimatesLoader(bz.data(events), columns)

class WithSplitAdjustedWindows(WithEstimateWindows):
    """
    ZiplineTestCase mixin providing fixures and a test to test running a
    Pipeline with an estimates loader over differently-sized windows and with
    split adjustments.
    """
    split_adjusted_asof_date = pd.Timestamp('2015-01-14')

    @classmethod
    def make_events(cls):
        if False:
            for i in range(10):
                print('nop')
        sid_30 = pd.DataFrame({TS_FIELD_NAME: [cls.window_test_start_date, pd.Timestamp('2015-01-09'), cls.window_test_start_date, pd.Timestamp('2015-01-20')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-20'), pd.Timestamp('2015-01-20')], 'estimate': [130.0, 131.0, 230.0, 231.0], FISCAL_QUARTER_FIELD_NAME: [1] * 2 + [2] * 2, FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 30})
        sid_40 = pd.DataFrame({TS_FIELD_NAME: [pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-15')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-09'), pd.Timestamp('2015-02-10')], 'estimate': [140.0, 240.0], FISCAL_QUARTER_FIELD_NAME: [1, 2], FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 40})
        sid_50 = pd.DataFrame({TS_FIELD_NAME: [pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-12')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-09'), pd.Timestamp('2015-02-10')], 'estimate': [150.0, 250.0], FISCAL_QUARTER_FIELD_NAME: [1, 2], FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 50})
        return pd.concat([cls.__base__.make_events(), sid_30, sid_40, sid_50])

    @classmethod
    def make_splits_data(cls):
        if False:
            i = 10
            return i + 15
        sid_0_splits = pd.DataFrame({SID_FIELD_NAME: 0, 'ratio': (-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100), 'effective_date': (pd.Timestamp('2014-01-01'), pd.Timestamp('2015-01-07'), pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-15'), pd.Timestamp('2015-01-18'), pd.Timestamp('2015-01-30'), pd.Timestamp('2016-01-01'))})
        sid_10_splits = pd.DataFrame({SID_FIELD_NAME: 10, 'ratio': (0.2, 0.3), 'effective_date': (pd.Timestamp('2015-01-07'), pd.Timestamp('2015-01-20'))})
        sid_20_splits = pd.DataFrame({SID_FIELD_NAME: 20, 'ratio': (0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 'effective_date': (pd.Timestamp('2015-01-07'), pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-15'), pd.Timestamp('2015-01-18'), pd.Timestamp('2015-01-30'))})
        sid_30_splits = pd.DataFrame({SID_FIELD_NAME: 30, 'ratio': (8, 9, 10, 11, 12), 'effective_date': (pd.Timestamp('2015-01-07'), pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-15'), pd.Timestamp('2015-01-18'))})
        sid_40_splits = pd.DataFrame({SID_FIELD_NAME: 40, 'ratio': (13, 14), 'effective_date': (pd.Timestamp('2015-01-20'), pd.Timestamp('2015-01-22'))})
        sid_50_splits = pd.DataFrame({SID_FIELD_NAME: 50, 'ratio': (15, 16), 'effective_date': (pd.Timestamp('2015-01-13'), pd.Timestamp('2015-01-14'))})
        return pd.concat([sid_0_splits, sid_10_splits, sid_20_splits, sid_30_splits, sid_40_splits, sid_50_splits])

class PreviousWithSplitAdjustedWindows(WithSplitAdjustedWindows, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            while True:
                i = 10
        return PreviousSplitAdjustedEarningsEstimatesLoader(events, columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate'], split_adjusted_asof=cls.split_adjusted_asof_date)

    @classmethod
    def make_expected_timelines(cls):
        if False:
            while True:
                i = 10
        oneq_previous = pd.concat([pd.concat([cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date), (30, 131 * 1 / 10, pd.Timestamp('2015-01-09')), (40, 140.0, pd.Timestamp('2015-01-09')), (50, 150 * 1 / 15 * 1 / 16, pd.Timestamp('2015-01-09'))], end_date) for end_date in pd.date_range('2015-01-09', '2015-01-12')]), cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date), (30, 131, pd.Timestamp('2015-01-09')), (40, 140.0, pd.Timestamp('2015-01-09')), (50, 150.0 * 1 / 16, pd.Timestamp('2015-01-09'))], pd.Timestamp('2015-01-13')), cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date), (30, 131, pd.Timestamp('2015-01-09')), (40, 140.0, pd.Timestamp('2015-01-09')), (50, 150.0, pd.Timestamp('2015-01-09'))], pd.Timestamp('2015-01-14')), pd.concat([cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date), (30, 131 * 11, pd.Timestamp('2015-01-09')), (40, 140.0, pd.Timestamp('2015-01-09')), (50, 150.0, pd.Timestamp('2015-01-09'))], end_date) for end_date in pd.date_range('2015-01-15', '2015-01-16')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 101, pd.Timestamp('2015-01-20')), (10, np.NaN, cls.window_test_start_date), (20, 121 * 0.7 * 0.8, pd.Timestamp('2015-01-20')), (30, 231, pd.Timestamp('2015-01-20')), (40, 140.0 * 13, pd.Timestamp('2015-01-09')), (50, 150.0, pd.Timestamp('2015-01-09'))], end_date) for end_date in pd.date_range('2015-01-20', '2015-01-21')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 101, pd.Timestamp('2015-01-20')), (10, 111 * 0.3, pd.Timestamp('2015-01-22')), (20, 121 * 0.7 * 0.8, pd.Timestamp('2015-01-20')), (30, 231, pd.Timestamp('2015-01-20')), (40, 140.0 * 13 * 14, pd.Timestamp('2015-01-09')), (50, 150.0, pd.Timestamp('2015-01-09'))], end_date) for end_date in pd.date_range('2015-01-22', '2015-01-29')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 101 * 7, pd.Timestamp('2015-01-20')), (10, 111 * 0.3, pd.Timestamp('2015-01-22')), (20, 121 * 0.7 * 0.8 * 0.9, pd.Timestamp('2015-01-20')), (30, 231, pd.Timestamp('2015-01-20')), (40, 140.0 * 13 * 14, pd.Timestamp('2015-01-09')), (50, 150.0, pd.Timestamp('2015-01-09'))], end_date) for end_date in pd.date_range('2015-01-30', '2015-02-04')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 101 * 7, pd.Timestamp('2015-01-20')), (10, 311 * 0.3, pd.Timestamp('2015-02-05')), (20, 121 * 0.7 * 0.8 * 0.9, pd.Timestamp('2015-01-20')), (30, 231, pd.Timestamp('2015-01-20')), (40, 140.0 * 13 * 14, pd.Timestamp('2015-01-09')), (50, 150.0, pd.Timestamp('2015-01-09'))], end_date) for end_date in pd.date_range('2015-02-05', '2015-02-09')]), cls.create_expected_df_for_factor_compute([(0, 201, pd.Timestamp('2015-02-10')), (10, 311 * 0.3, pd.Timestamp('2015-02-05')), (20, 221 * 0.8 * 0.9, pd.Timestamp('2015-02-10')), (30, 231, pd.Timestamp('2015-01-20')), (40, 240.0 * 13 * 14, pd.Timestamp('2015-02-10')), (50, 250.0, pd.Timestamp('2015-02-10'))], pd.Timestamp('2015-02-10'))])
        twoq_previous = pd.concat([cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date), (30, np.NaN, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-09', '2015-01-19')] + [cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date), (30, 131 * 11 * 12, pd.Timestamp('2015-01-20'))], end_date) for end_date in pd.date_range('2015-01-20', '2015-02-09')] + [cls.create_expected_df_for_factor_compute([(0, 101 * 7, pd.Timestamp('2015-02-10')), (10, np.NaN, pd.Timestamp('2015-02-05')), (20, 121 * 0.7 * 0.8 * 0.9, pd.Timestamp('2015-02-10')), (30, 131 * 11 * 12, pd.Timestamp('2015-01-20')), (40, 140.0 * 13 * 14, pd.Timestamp('2015-02-10')), (50, 150.0, pd.Timestamp('2015-02-10'))], pd.Timestamp('2015-02-10'))])
        return {1: oneq_previous, 2: twoq_previous}

class BlazePreviousWithSplitAdjustedWindows(PreviousWithSplitAdjustedWindows):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return BlazePreviousSplitAdjustedEstimatesLoader(bz.data(events), columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate'], split_adjusted_asof=cls.split_adjusted_asof_date)

class NextWithSplitAdjustedWindows(WithSplitAdjustedWindows, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return NextSplitAdjustedEarningsEstimatesLoader(events, columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate'], split_adjusted_asof=cls.split_adjusted_asof_date)

    @classmethod
    def make_expected_timelines(cls):
        if False:
            i = 10
            return i + 15
        oneq_next = pd.concat([cls.create_expected_df_for_factor_compute([(0, 100 * 1 / 4, cls.window_test_start_date), (10, 110, pd.Timestamp('2015-01-09')), (20, 120 * 5 / 3, cls.window_test_start_date), (20, 121 * 5 / 3, pd.Timestamp('2015-01-07')), (30, 130 * 1 / 10, cls.window_test_start_date), (30, 131 * 1 / 10, pd.Timestamp('2015-01-09')), (40, 140, pd.Timestamp('2015-01-09')), (50, 150.0 * 1 / 15 * 1 / 16, pd.Timestamp('2015-01-09'))], pd.Timestamp('2015-01-09')), cls.create_expected_df_for_factor_compute([(0, 100 * 1 / 4, cls.window_test_start_date), (10, 110, pd.Timestamp('2015-01-09')), (10, 111, pd.Timestamp('2015-01-12')), (20, 120 * 5 / 3, cls.window_test_start_date), (20, 121 * 5 / 3, pd.Timestamp('2015-01-07')), (30, 230 * 1 / 10, cls.window_test_start_date), (40, np.NaN, pd.Timestamp('2015-01-10')), (50, 250.0 * 1 / 15 * 1 / 16, pd.Timestamp('2015-01-12'))], pd.Timestamp('2015-01-12')), cls.create_expected_df_for_factor_compute([(0, 100, cls.window_test_start_date), (10, 110, pd.Timestamp('2015-01-09')), (10, 111, pd.Timestamp('2015-01-12')), (20, 120, cls.window_test_start_date), (20, 121, pd.Timestamp('2015-01-07')), (30, 230, cls.window_test_start_date), (40, np.NaN, pd.Timestamp('2015-01-10')), (50, 250.0 * 1 / 16, pd.Timestamp('2015-01-12'))], pd.Timestamp('2015-01-13')), cls.create_expected_df_for_factor_compute([(0, 100, cls.window_test_start_date), (10, 110, pd.Timestamp('2015-01-09')), (10, 111, pd.Timestamp('2015-01-12')), (20, 120, cls.window_test_start_date), (20, 121, pd.Timestamp('2015-01-07')), (30, 230, cls.window_test_start_date), (40, np.NaN, pd.Timestamp('2015-01-10')), (50, 250.0, pd.Timestamp('2015-01-12'))], pd.Timestamp('2015-01-14')), pd.concat([cls.create_expected_df_for_factor_compute([(0, 100 * 5, cls.window_test_start_date), (10, 110, pd.Timestamp('2015-01-09')), (10, 111, pd.Timestamp('2015-01-12')), (20, 120 * 0.7, cls.window_test_start_date), (20, 121 * 0.7, pd.Timestamp('2015-01-07')), (30, 230 * 11, cls.window_test_start_date), (40, 240, pd.Timestamp('2015-01-15')), (50, 250.0, pd.Timestamp('2015-01-12'))], end_date) for end_date in pd.date_range('2015-01-15', '2015-01-16')]), cls.create_expected_df_for_factor_compute([(0, 100 * 5 * 6, cls.window_test_start_date), (0, 101, pd.Timestamp('2015-01-20')), (10, 110 * 0.3, pd.Timestamp('2015-01-09')), (10, 111 * 0.3, pd.Timestamp('2015-01-12')), (20, 120 * 0.7 * 0.8, cls.window_test_start_date), (20, 121 * 0.7 * 0.8, pd.Timestamp('2015-01-07')), (30, 230 * 11 * 12, cls.window_test_start_date), (30, 231, pd.Timestamp('2015-01-20')), (40, 240 * 13, pd.Timestamp('2015-01-15')), (50, 250.0, pd.Timestamp('2015-01-12'))], pd.Timestamp('2015-01-20')), cls.create_expected_df_for_factor_compute([(0, 200 * 5 * 6, pd.Timestamp('2015-01-12')), (10, 110 * 0.3, pd.Timestamp('2015-01-09')), (10, 111 * 0.3, pd.Timestamp('2015-01-12')), (20, 220 * 0.7 * 0.8, cls.window_test_start_date), (20, 221 * 0.8, pd.Timestamp('2015-01-17')), (40, 240 * 13, pd.Timestamp('2015-01-15')), (50, 250.0, pd.Timestamp('2015-01-12'))], pd.Timestamp('2015-01-21')), cls.create_expected_df_for_factor_compute([(0, 200 * 5 * 6, pd.Timestamp('2015-01-12')), (10, 110 * 0.3, pd.Timestamp('2015-01-09')), (10, 111 * 0.3, pd.Timestamp('2015-01-12')), (20, 220 * 0.7 * 0.8, cls.window_test_start_date), (20, 221 * 0.8, pd.Timestamp('2015-01-17')), (40, 240 * 13 * 14, pd.Timestamp('2015-01-15')), (50, 250.0, pd.Timestamp('2015-01-12'))], pd.Timestamp('2015-01-22')), pd.concat([cls.create_expected_df_for_factor_compute([(0, 200 * 5 * 6, pd.Timestamp('2015-01-12')), (10, 310 * 0.3, pd.Timestamp('2015-01-09')), (10, 311 * 0.3, pd.Timestamp('2015-01-15')), (20, 220 * 0.7 * 0.8, cls.window_test_start_date), (20, 221 * 0.8, pd.Timestamp('2015-01-17')), (40, 240 * 13 * 14, pd.Timestamp('2015-01-15')), (50, 250.0, pd.Timestamp('2015-01-12'))], end_date) for end_date in pd.date_range('2015-01-23', '2015-01-29')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 200 * 5 * 6 * 7, pd.Timestamp('2015-01-12')), (10, 310 * 0.3, pd.Timestamp('2015-01-09')), (10, 311 * 0.3, pd.Timestamp('2015-01-15')), (20, 220 * 0.7 * 0.8 * 0.9, cls.window_test_start_date), (20, 221 * 0.8 * 0.9, pd.Timestamp('2015-01-17')), (40, 240 * 13 * 14, pd.Timestamp('2015-01-15')), (50, 250.0, pd.Timestamp('2015-01-12'))], end_date) for end_date in pd.date_range('2015-01-30', '2015-02-05')]), pd.concat([cls.create_expected_df_for_factor_compute([(0, 200 * 5 * 6 * 7, pd.Timestamp('2015-01-12')), (10, np.NaN, cls.window_test_start_date), (20, 220 * 0.7 * 0.8 * 0.9, cls.window_test_start_date), (20, 221 * 0.8 * 0.9, pd.Timestamp('2015-01-17')), (40, 240 * 13 * 14, pd.Timestamp('2015-01-15')), (50, 250.0, pd.Timestamp('2015-01-12'))], end_date) for end_date in pd.date_range('2015-02-06', '2015-02-09')]), cls.create_expected_df_for_factor_compute([(0, 200 * 5 * 6 * 7, pd.Timestamp('2015-01-12')), (0, 201, pd.Timestamp('2015-02-10')), (10, np.NaN, cls.window_test_start_date), (20, 220 * 0.7 * 0.8 * 0.9, cls.window_test_start_date), (20, 221 * 0.8 * 0.9, pd.Timestamp('2015-01-17')), (40, 240 * 13 * 14, pd.Timestamp('2015-01-15')), (50, 250.0, pd.Timestamp('2015-01-12'))], pd.Timestamp('2015-02-10'))])
        twoq_next = pd.concat([cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, 220 * 5 / 3, cls.window_test_start_date), (30, 230 * 1 / 10, cls.window_test_start_date), (40, np.NaN, cls.window_test_start_date), (50, np.NaN, cls.window_test_start_date)], pd.Timestamp('2015-01-09'))] + [cls.create_expected_df_for_factor_compute([(0, 200 * 1 / 4, pd.Timestamp('2015-01-12')), (10, np.NaN, cls.window_test_start_date), (20, 220 * 5 / 3, cls.window_test_start_date), (30, np.NaN, cls.window_test_start_date), (40, np.NaN, cls.window_test_start_date)], pd.Timestamp('2015-01-12'))] + [cls.create_expected_df_for_factor_compute([(0, 200, pd.Timestamp('2015-01-12')), (10, np.NaN, cls.window_test_start_date), (20, 220, cls.window_test_start_date), (30, np.NaN, cls.window_test_start_date), (40, np.NaN, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-13', '2015-01-14')] + [cls.create_expected_df_for_factor_compute([(0, 200 * 5, pd.Timestamp('2015-01-12')), (10, np.NaN, cls.window_test_start_date), (20, 220 * 0.7, cls.window_test_start_date), (30, np.NaN, cls.window_test_start_date), (40, np.NaN, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-15', '2015-01-16')] + [cls.create_expected_df_for_factor_compute([(0, 200 * 5 * 6, pd.Timestamp('2015-01-12')), (10, np.NaN, cls.window_test_start_date), (20, 220 * 0.7 * 0.8, cls.window_test_start_date), (20, 221 * 0.8, pd.Timestamp('2015-01-17')), (30, np.NaN, cls.window_test_start_date), (40, np.NaN, cls.window_test_start_date)], pd.Timestamp('2015-01-20'))] + [cls.create_expected_df_for_factor_compute([(0, np.NaN, cls.window_test_start_date), (10, np.NaN, cls.window_test_start_date), (20, np.NaN, cls.window_test_start_date), (30, np.NaN, cls.window_test_start_date), (40, np.NaN, cls.window_test_start_date)], end_date) for end_date in pd.date_range('2015-01-21', '2015-02-10')])
        return {1: oneq_next, 2: twoq_next}

class BlazeNextWithSplitAdjustedWindows(NextWithSplitAdjustedWindows):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return BlazeNextSplitAdjustedEstimatesLoader(bz.data(events), columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate'], split_adjusted_asof=cls.split_adjusted_asof_date)

class WithSplitAdjustedMultipleEstimateColumns(WithEstimates):
    """
    ZiplineTestCase mixin for having multiple estimate columns that are
    split-adjusted to make sure that adjustments are applied correctly.

    Attributes
    ----------
    test_start_date : pd.Timestamp
        The start date of the test.
    test_end_date : pd.Timestamp
        The start date of the test.
    split_adjusted_asof : pd.Timestamp
        The split-adjusted-asof-date of the data used in the test, to be used
        to create all loaders of test classes that subclass this mixin.

    Methods
    -------
    make_expected_timelines_1q_out -> dict[pd.Timestamp -> dict[str ->
        np.array]]
        The expected array of results for each date of the date range for
        each column. Only for 1 quarter out.

    make_expected_timelines_2q_out -> dict[pd.Timestamp -> dict[str ->
        np.array]]
        The expected array of results for each date of the date range. For 2
        quarters out, so only for the column that is requested to be loaded
        with 2 quarters out.

    Tests
    -----
    test_adjustments_with_multiple_adjusted_columns
        Tests that if you have multiple columns, we still split-adjust
        correctly.

    test_multiple_datasets_different_num_announcements
        Tests that if you have multiple datasets that ask for a different
        number of quarters out, and each asks for a different estimates column,
        we still split-adjust correctly.
    """
    END_DATE = pd.Timestamp('2015-02-10')
    test_start_date = pd.Timestamp('2015-01-06', tz='utc')
    test_end_date = pd.Timestamp('2015-01-12', tz='utc')
    split_adjusted_asof = pd.Timestamp('2015-01-08')

    @classmethod
    def make_columns(cls):
        if False:
            i = 10
            return i + 15
        return {MultipleColumnsEstimates.event_date: 'event_date', MultipleColumnsEstimates.fiscal_quarter: 'fiscal_quarter', MultipleColumnsEstimates.fiscal_year: 'fiscal_year', MultipleColumnsEstimates.estimate1: 'estimate1', MultipleColumnsEstimates.estimate2: 'estimate2'}

    @classmethod
    def make_events(cls):
        if False:
            print('Hello World!')
        sid_0_events = pd.DataFrame({TS_FIELD_NAME: [pd.Timestamp('2015-01-05'), pd.Timestamp('2015-01-05')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-09'), pd.Timestamp('2015-01-12')], 'estimate1': [1100.0, 1200.0], 'estimate2': [2100.0, 2200.0], FISCAL_QUARTER_FIELD_NAME: [1, 2], FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 0})
        sid_1_events = pd.DataFrame({TS_FIELD_NAME: [pd.Timestamp('2015-01-05'), pd.Timestamp('2015-01-05')], EVENT_DATE_FIELD_NAME: [pd.Timestamp('2015-01-08'), pd.Timestamp('2015-01-11')], 'estimate1': [1110.0, 1210.0], 'estimate2': [2110.0, 2210.0], FISCAL_QUARTER_FIELD_NAME: [1, 2], FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 1})
        return pd.concat([sid_0_events, sid_1_events])

    @classmethod
    def make_splits_data(cls):
        if False:
            while True:
                i = 10
        sid_0_splits = pd.DataFrame({SID_FIELD_NAME: 0, 'ratio': (0.3, 3.0), 'effective_date': (pd.Timestamp('2015-01-07'), pd.Timestamp('2015-01-09'))})
        sid_1_splits = pd.DataFrame({SID_FIELD_NAME: 1, 'ratio': (0.4, 4.0), 'effective_date': (pd.Timestamp('2015-01-07'), pd.Timestamp('2015-01-09'))})
        return pd.concat([sid_0_splits, sid_1_splits])

    @classmethod
    def make_expected_timelines_1q_out(cls):
        if False:
            return 10
        return {}

    @classmethod
    def make_expected_timelines_2q_out(cls):
        if False:
            while True:
                i = 10
        return {}

    @classmethod
    def init_class_fixtures(cls):
        if False:
            for i in range(10):
                print('nop')
        super(WithSplitAdjustedMultipleEstimateColumns, cls).init_class_fixtures()
        cls.timelines_1q_out = cls.make_expected_timelines_1q_out()
        cls.timelines_2q_out = cls.make_expected_timelines_2q_out()

    def test_adjustments_with_multiple_adjusted_columns(self):
        if False:
            print('Hello World!')
        dataset = MultipleColumnsQuartersEstimates(1)
        timelines = self.timelines_1q_out
        window_len = 3

        class SomeFactor(CustomFactor):
            inputs = [dataset.estimate1, dataset.estimate2]
            window_length = window_len

            def compute(self, today, assets, out, estimate1, estimate2):
                if False:
                    while True:
                        i = 10
                assert_almost_equal(estimate1, timelines[today]['estimate1'])
                assert_almost_equal(estimate2, timelines[today]['estimate2'])
        engine = self.make_engine()
        engine.run_pipeline(Pipeline({'est': SomeFactor()}), start_date=self.test_start_date, end_date=self.test_end_date)

    def test_multiple_datasets_different_num_announcements(self):
        if False:
            for i in range(10):
                print('nop')
        dataset1 = MultipleColumnsQuartersEstimates(1)
        dataset2 = MultipleColumnsQuartersEstimates(2)
        timelines_1q_out = self.timelines_1q_out
        timelines_2q_out = self.timelines_2q_out
        window_len = 3

        class SomeFactor1(CustomFactor):
            inputs = [dataset1.estimate1]
            window_length = window_len

            def compute(self, today, assets, out, estimate1):
                if False:
                    for i in range(10):
                        print('nop')
                assert_almost_equal(estimate1, timelines_1q_out[today]['estimate1'])

        class SomeFactor2(CustomFactor):
            inputs = [dataset2.estimate2]
            window_length = window_len

            def compute(self, today, assets, out, estimate2):
                if False:
                    i = 10
                    return i + 15
                assert_almost_equal(estimate2, timelines_2q_out[today]['estimate2'])
        engine = self.make_engine()
        engine.run_pipeline(Pipeline({'est1': SomeFactor1(), 'est2': SomeFactor2()}), start_date=self.test_start_date, end_date=self.test_end_date)

class PreviousWithSplitAdjustedMultipleEstimateColumns(WithSplitAdjustedMultipleEstimateColumns, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            for i in range(10):
                print('nop')
        return PreviousSplitAdjustedEarningsEstimatesLoader(events, columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate1', 'estimate2'], split_adjusted_asof=cls.split_adjusted_asof)

    @classmethod
    def make_expected_timelines_1q_out(cls):
        if False:
            while True:
                i = 10
        return {pd.Timestamp('2015-01-06', tz='utc'): {'estimate1': np.array([[np.NaN, np.NaN]] * 3), 'estimate2': np.array([[np.NaN, np.NaN]] * 3)}, pd.Timestamp('2015-01-07', tz='utc'): {'estimate1': np.array([[np.NaN, np.NaN]] * 3), 'estimate2': np.array([[np.NaN, np.NaN]] * 3)}, pd.Timestamp('2015-01-08', tz='utc'): {'estimate1': np.array([[np.NaN, np.NaN]] * 2 + [[np.NaN, 1110.0]]), 'estimate2': np.array([[np.NaN, np.NaN]] * 2 + [[np.NaN, 2110.0]])}, pd.Timestamp('2015-01-09', tz='utc'): {'estimate1': np.array([[np.NaN, np.NaN]] + [[np.NaN, 1110.0 * 4]] + [[1100 * 3.0, 1110.0 * 4]]), 'estimate2': np.array([[np.NaN, np.NaN]] + [[np.NaN, 2110.0 * 4]] + [[2100 * 3.0, 2110.0 * 4]])}, pd.Timestamp('2015-01-12', tz='utc'): {'estimate1': np.array([[np.NaN, np.NaN]] * 2 + [[1200 * 3.0, 1210.0 * 4]]), 'estimate2': np.array([[np.NaN, np.NaN]] * 2 + [[2200 * 3.0, 2210.0 * 4]])}}

    @classmethod
    def make_expected_timelines_2q_out(cls):
        if False:
            print('Hello World!')
        return {pd.Timestamp('2015-01-06', tz='utc'): {'estimate2': np.array([[np.NaN, np.NaN]] * 3)}, pd.Timestamp('2015-01-07', tz='utc'): {'estimate2': np.array([[np.NaN, np.NaN]] * 3)}, pd.Timestamp('2015-01-08', tz='utc'): {'estimate2': np.array([[np.NaN, np.NaN]] * 3)}, pd.Timestamp('2015-01-09', tz='utc'): {'estimate2': np.array([[np.NaN, np.NaN]] * 3)}, pd.Timestamp('2015-01-12', tz='utc'): {'estimate2': np.array([[np.NaN, np.NaN]] * 2 + [[2100 * 3.0, 2110.0 * 4]])}}

class BlazePreviousWithMultipleEstimateColumns(PreviousWithSplitAdjustedMultipleEstimateColumns):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            while True:
                i = 10
        return BlazePreviousSplitAdjustedEstimatesLoader(bz.data(events), columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate1', 'estimate2'], split_adjusted_asof=cls.split_adjusted_asof)

class NextWithSplitAdjustedMultipleEstimateColumns(WithSplitAdjustedMultipleEstimateColumns, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            print('Hello World!')
        return NextSplitAdjustedEarningsEstimatesLoader(events, columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate1', 'estimate2'], split_adjusted_asof=cls.split_adjusted_asof)

    @classmethod
    def make_expected_timelines_1q_out(cls):
        if False:
            while True:
                i = 10
        return {pd.Timestamp('2015-01-06', tz='utc'): {'estimate1': np.array([[np.NaN, np.NaN]] + [[1100.0 * 1 / 0.3, 1110.0 * 1 / 0.4]] * 2), 'estimate2': np.array([[np.NaN, np.NaN]] + [[2100.0 * 1 / 0.3, 2110.0 * 1 / 0.4]] * 2)}, pd.Timestamp('2015-01-07', tz='utc'): {'estimate1': np.array([[1100.0, 1110.0]] * 3), 'estimate2': np.array([[2100.0, 2110.0]] * 3)}, pd.Timestamp('2015-01-08', tz='utc'): {'estimate1': np.array([[1100.0, 1110.0]] * 3), 'estimate2': np.array([[2100.0, 2110.0]] * 3)}, pd.Timestamp('2015-01-09', tz='utc'): {'estimate1': np.array([[1100 * 3.0, 1210.0 * 4]] * 3), 'estimate2': np.array([[2100 * 3.0, 2210.0 * 4]] * 3)}, pd.Timestamp('2015-01-12', tz='utc'): {'estimate1': np.array([[1200 * 3.0, np.NaN]] * 3), 'estimate2': np.array([[2200 * 3.0, np.NaN]] * 3)}}

    @classmethod
    def make_expected_timelines_2q_out(cls):
        if False:
            print('Hello World!')
        return {pd.Timestamp('2015-01-06', tz='utc'): {'estimate2': np.array([[np.NaN, np.NaN]] + [[2200 * 1 / 0.3, 2210.0 * 1 / 0.4]] * 2)}, pd.Timestamp('2015-01-07', tz='utc'): {'estimate2': np.array([[2200.0, 2210.0]] * 3)}, pd.Timestamp('2015-01-08', tz='utc'): {'estimate2': np.array([[2200, 2210.0]] * 3)}, pd.Timestamp('2015-01-09', tz='utc'): {'estimate2': np.array([[2200 * 3.0, np.NaN]] * 3)}, pd.Timestamp('2015-01-12', tz='utc'): {'estimate2': np.array([[np.NaN, np.NaN]] * 3)}}

class BlazeNextWithMultipleEstimateColumns(NextWithSplitAdjustedMultipleEstimateColumns):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return BlazeNextSplitAdjustedEstimatesLoader(bz.data(events), columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate1', 'estimate2'], split_adjusted_asof=cls.split_adjusted_asof)

class WithAdjustmentBoundaries(WithEstimates):
    """
    ZiplineTestCase mixin providing class-level attributes, methods,
    and a test to make sure that when the split-adjusted-asof-date is not
    strictly within the date index, we can still apply adjustments correctly.

    Attributes
    ----------
    split_adjusted_before_start : pd.Timestamp
        A split-adjusted-asof-date before the start date of the test.
    split_adjusted_after_end : pd.Timestamp
        A split-adjusted-asof-date before the end date of the test.
    split_adjusted_asof_dates : list of tuples of pd.Timestamp
        All the split-adjusted-asof-dates over which we want to parameterize
        the test.

    Methods
    -------
    make_expected_out -> dict[pd.Timestamp -> pd.DataFrame]
        A dictionary of the expected output of the pipeline at each of the
        dates of interest.
    """
    START_DATE = pd.Timestamp('2015-01-04')
    test_start_date = pd.Timestamp('2015-01-05')
    END_DATE = test_end_date = pd.Timestamp('2015-01-12')
    split_adjusted_before_start = test_start_date - timedelta(days=1)
    split_adjusted_after_end = test_end_date + timedelta(days=1)
    split_adjusted_asof_dates = [(test_start_date,), (test_end_date,), (split_adjusted_before_start,), (split_adjusted_after_end,)]

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(WithAdjustmentBoundaries, cls).init_class_fixtures()
        cls.s0 = cls.asset_finder.retrieve_asset(0)
        cls.s1 = cls.asset_finder.retrieve_asset(1)
        cls.s2 = cls.asset_finder.retrieve_asset(2)
        cls.s3 = cls.asset_finder.retrieve_asset(3)
        cls.s4 = cls.asset_finder.retrieve_asset(4)
        cls.expected = cls.make_expected_out()

    @classmethod
    def make_events(cls):
        if False:
            print('Hello World!')
        sid_0_timeline = pd.DataFrame({TS_FIELD_NAME: cls.test_start_date, EVENT_DATE_FIELD_NAME: pd.Timestamp('2015-01-09'), 'estimate': 10.0, FISCAL_QUARTER_FIELD_NAME: 1, FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 0}, index=[0])
        sid_1_timeline = pd.DataFrame({TS_FIELD_NAME: cls.test_start_date, EVENT_DATE_FIELD_NAME: cls.test_start_date, 'estimate': 11.0, FISCAL_QUARTER_FIELD_NAME: 1, FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 1}, index=[0])
        sid_2_timeline = pd.DataFrame({TS_FIELD_NAME: cls.test_end_date, EVENT_DATE_FIELD_NAME: cls.test_end_date + timedelta(days=1), 'estimate': 12.0, FISCAL_QUARTER_FIELD_NAME: 1, FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 2}, index=[0])
        sid_3_timeline = pd.DataFrame({TS_FIELD_NAME: cls.test_end_date - timedelta(days=1), EVENT_DATE_FIELD_NAME: cls.test_end_date, 'estimate': 13.0, FISCAL_QUARTER_FIELD_NAME: 1, FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 3}, index=[0])
        sid_4_timeline = pd.DataFrame({TS_FIELD_NAME: cls.test_end_date - timedelta(days=1), EVENT_DATE_FIELD_NAME: cls.test_end_date - timedelta(days=1), 'estimate': 14.0, FISCAL_QUARTER_FIELD_NAME: 1, FISCAL_YEAR_FIELD_NAME: 2015, SID_FIELD_NAME: 4}, index=[0])
        return pd.concat([sid_0_timeline, sid_1_timeline, sid_2_timeline, sid_3_timeline, sid_4_timeline])

    @classmethod
    def make_splits_data(cls):
        if False:
            for i in range(10):
                print('nop')
        sid_0_splits = pd.DataFrame({SID_FIELD_NAME: 0, 'ratio': 0.1, 'effective_date': cls.test_start_date}, index=[0])
        sid_1_splits = pd.DataFrame({SID_FIELD_NAME: 1, 'ratio': 0.11, 'effective_date': cls.test_start_date}, index=[0])
        sid_2_splits = pd.DataFrame({SID_FIELD_NAME: 2, 'ratio': 0.12, 'effective_date': cls.test_end_date}, index=[0])
        sid_3_splits = pd.DataFrame({SID_FIELD_NAME: 3, 'ratio': 0.13, 'effective_date': cls.test_end_date}, index=[0])
        sid_4_splits = pd.DataFrame({SID_FIELD_NAME: 4, 'ratio': (0.14, 0.15), 'effective_date': (cls.test_start_date, cls.test_end_date)})
        return pd.concat([sid_0_splits, sid_1_splits, sid_2_splits, sid_3_splits, sid_4_splits])

    @parameterized.expand(split_adjusted_asof_dates)
    def test_boundaries(self, split_date):
        if False:
            print('Hello World!')
        dataset = QuartersEstimates(1)
        loader = self.loader(split_adjusted_asof=split_date)
        engine = engine = self.make_engine(loader)
        result = engine.run_pipeline(Pipeline({'estimate': dataset.estimate.latest}), start_date=self.trading_days[0], end_date=self.trading_days[-1])
        expected = self.expected[split_date]
        assert_frame_equal(result, expected, check_names=False)

    @classmethod
    def make_expected_out(cls):
        if False:
            print('Hello World!')
        return {}

class PreviousWithAdjustmentBoundaries(WithAdjustmentBoundaries, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            while True:
                i = 10
        return partial(PreviousSplitAdjustedEarningsEstimatesLoader, events, columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate'])

    @classmethod
    def make_expected_out(cls):
        if False:
            return 10
        split_adjusted_at_start_boundary = pd.concat([pd.DataFrame({SID_FIELD_NAME: cls.s0, 'estimate': np.NaN}, index=pd.date_range(cls.test_start_date, pd.Timestamp('2015-01-08'), tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s0, 'estimate': 10.0}, index=pd.date_range(pd.Timestamp('2015-01-09'), cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s1, 'estimate': 11.0}, index=pd.date_range(cls.test_start_date, cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s2, 'estimate': np.NaN}, index=pd.date_range(cls.test_start_date, cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s3, 'estimate': np.NaN}, index=pd.date_range(cls.test_start_date, cls.test_end_date - timedelta(1), tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s3, 'estimate': 13.0 * 0.13}, index=pd.date_range(cls.test_end_date, cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s4, 'estimate': np.NaN}, index=pd.date_range(cls.test_start_date, cls.test_end_date - timedelta(2), tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s4, 'estimate': 14.0 * 0.15}, index=pd.date_range(cls.test_end_date - timedelta(1), cls.test_end_date, tz='utc'))]).set_index(SID_FIELD_NAME, append=True).unstack(SID_FIELD_NAME).reindex(cls.trading_days).stack(SID_FIELD_NAME, dropna=False)
        split_adjusted_at_end_boundary = pd.concat([pd.DataFrame({SID_FIELD_NAME: cls.s0, 'estimate': np.NaN}, index=pd.date_range(cls.test_start_date, pd.Timestamp('2015-01-08'), tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s0, 'estimate': 10.0}, index=pd.date_range(pd.Timestamp('2015-01-09'), cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s1, 'estimate': 11.0}, index=pd.date_range(cls.test_start_date, cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s2, 'estimate': np.NaN}, index=pd.date_range(cls.test_start_date, cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s3, 'estimate': np.NaN}, index=pd.date_range(cls.test_start_date, cls.test_end_date - timedelta(1), tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s3, 'estimate': 13.0}, index=pd.date_range(cls.test_end_date, cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s4, 'estimate': np.NaN}, index=pd.date_range(cls.test_start_date, cls.test_end_date - timedelta(2), tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s4, 'estimate': 14.0}, index=pd.date_range(cls.test_end_date - timedelta(1), cls.test_end_date, tz='utc'))]).set_index(SID_FIELD_NAME, append=True).unstack(SID_FIELD_NAME).reindex(cls.trading_days).stack(SID_FIELD_NAME, dropna=False)
        split_adjusted_before_start_boundary = split_adjusted_at_start_boundary
        split_adjusted_after_end_boundary = split_adjusted_at_end_boundary
        return {cls.test_start_date: split_adjusted_at_start_boundary, cls.split_adjusted_before_start: split_adjusted_before_start_boundary, cls.test_end_date: split_adjusted_at_end_boundary, cls.split_adjusted_after_end: split_adjusted_after_end_boundary}

class BlazePreviousWithAdjustmentBoundaries(PreviousWithAdjustmentBoundaries):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            while True:
                i = 10
        return partial(BlazePreviousSplitAdjustedEstimatesLoader, bz.data(events), columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate'])

class NextWithAdjustmentBoundaries(WithAdjustmentBoundaries, ZiplineTestCase):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return partial(NextSplitAdjustedEarningsEstimatesLoader, events, columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate'])

    @classmethod
    def make_expected_out(cls):
        if False:
            for i in range(10):
                print('nop')
        split_adjusted_at_start_boundary = pd.concat([pd.DataFrame({SID_FIELD_NAME: cls.s0, 'estimate': 10}, index=pd.date_range(cls.test_start_date, pd.Timestamp('2015-01-09'), tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s1, 'estimate': 11.0}, index=pd.date_range(cls.test_start_date, cls.test_start_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s2, 'estimate': 12.0}, index=pd.date_range(cls.test_end_date, cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s3, 'estimate': 13.0 * 0.13}, index=pd.date_range(cls.test_end_date - timedelta(1), cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s4, 'estimate': 14.0}, index=pd.date_range(cls.test_end_date - timedelta(1), cls.test_end_date - timedelta(1), tz='utc'))]).set_index(SID_FIELD_NAME, append=True).unstack(SID_FIELD_NAME).reindex(cls.trading_days).stack(SID_FIELD_NAME, dropna=False)
        split_adjusted_at_end_boundary = pd.concat([pd.DataFrame({SID_FIELD_NAME: cls.s0, 'estimate': 10}, index=pd.date_range(cls.test_start_date, pd.Timestamp('2015-01-09'), tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s1, 'estimate': 11.0}, index=pd.date_range(cls.test_start_date, cls.test_start_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s2, 'estimate': 12.0}, index=pd.date_range(cls.test_end_date, cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s3, 'estimate': 13.0}, index=pd.date_range(cls.test_end_date - timedelta(1), cls.test_end_date, tz='utc')), pd.DataFrame({SID_FIELD_NAME: cls.s4, 'estimate': 14.0}, index=pd.date_range(cls.test_end_date - timedelta(1), cls.test_end_date - timedelta(1), tz='utc'))]).set_index(SID_FIELD_NAME, append=True).unstack(SID_FIELD_NAME).reindex(cls.trading_days).stack(SID_FIELD_NAME, dropna=False)
        split_adjusted_before_start_boundary = split_adjusted_at_start_boundary
        split_adjusted_after_end_boundary = split_adjusted_at_end_boundary
        return {cls.test_start_date: split_adjusted_at_start_boundary, cls.split_adjusted_before_start: split_adjusted_before_start_boundary, cls.test_end_date: split_adjusted_at_end_boundary, cls.split_adjusted_after_end: split_adjusted_after_end_boundary}

class BlazeNextWithAdjustmentBoundaries(NextWithAdjustmentBoundaries):

    @classmethod
    def make_loader(cls, events, columns):
        if False:
            return 10
        return partial(BlazeNextSplitAdjustedEstimatesLoader, bz.data(events), columns, split_adjustments_loader=cls.adjustment_reader, split_adjusted_column_names=['estimate'])

class QuarterShiftTestCase(ZiplineTestCase):
    """
    This tests, in isolation, quarter calculation logic for shifting quarters
    backwards/forwards from a starting point.
    """

    def test_quarter_normalization(self):
        if False:
            return 10
        input_yrs = pd.Series(range(2011, 2015), dtype=np.int64)
        input_qtrs = pd.Series(range(1, 5), dtype=np.int64)
        (result_years, result_quarters) = split_normalized_quarters(normalize_quarters(input_yrs, input_qtrs))
        assert_equal(input_yrs, result_years)
        assert_equal(input_qtrs, result_quarters)