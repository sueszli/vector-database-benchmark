from abc import abstractmethod, abstractproperty
from interface import implements
import numpy as np
import pandas as pd
from six import viewvalues
from toolz import groupby
from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.adjustment import Datetime641DArrayOverwrite, Datetime64Overwrite, Float641DArrayOverwrite, Float64Multiply, Float64Overwrite
from zipline.pipeline.common import EVENT_DATE_FIELD_NAME, FISCAL_QUARTER_FIELD_NAME, FISCAL_YEAR_FIELD_NAME, SID_FIELD_NAME, TS_FIELD_NAME
from zipline.pipeline.loaders.base import PipelineLoader
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype
from zipline.pipeline.loaders.utils import ffill_across_cols, last_in_date_group
INVALID_NUM_QTRS_MESSAGE = 'Passed invalid number of quarters %s; must pass a number of quarters >= 0'
NEXT_FISCAL_QUARTER = 'next_fiscal_quarter'
NEXT_FISCAL_YEAR = 'next_fiscal_year'
NORMALIZED_QUARTERS = 'normalized_quarters'
PREVIOUS_FISCAL_QUARTER = 'previous_fiscal_quarter'
PREVIOUS_FISCAL_YEAR = 'previous_fiscal_year'
SHIFTED_NORMALIZED_QTRS = 'shifted_normalized_quarters'
SIMULATION_DATES = 'dates'

def normalize_quarters(years, quarters):
    if False:
        return 10
    return years * 4 + quarters - 1

def split_normalized_quarters(normalized_quarters):
    if False:
        return 10
    years = normalized_quarters // 4
    quarters = normalized_quarters % 4
    return (years, quarters + 1)
metadata_columns = frozenset({TS_FIELD_NAME, SID_FIELD_NAME, EVENT_DATE_FIELD_NAME, FISCAL_QUARTER_FIELD_NAME, FISCAL_YEAR_FIELD_NAME})

def required_estimates_fields(columns):
    if False:
        i = 10
        return i + 15
    '\n    Compute the set of resource columns required to serve\n    `columns`.\n    '
    return metadata_columns.union(viewvalues(columns))

def validate_column_specs(events, columns):
    if False:
        while True:
            i = 10
    '\n    Verify that the columns of ``events`` can be used by a\n    EarningsEstimatesLoader to serve the BoundColumns described by\n    `columns`.\n    '
    required = required_estimates_fields(columns)
    received = set(events.columns)
    missing = required - received
    if missing:
        raise ValueError('EarningsEstimatesLoader missing required columns {missing}.\nGot Columns: {received}\nExpected Columns: {required}'.format(missing=sorted(missing), received=sorted(received), required=sorted(required)))

def add_new_adjustments(adjustments_dict, adjustments, column_name, ts):
    if False:
        return 10
    try:
        adjustments_dict[column_name][ts].extend(adjustments)
    except KeyError:
        adjustments_dict[column_name][ts] = adjustments

class EarningsEstimatesLoader(implements(PipelineLoader)):
    """
    An abstract pipeline loader for estimates data that can load data a
    variable number of quarters forwards/backwards from calendar dates
    depending on the `num_announcements` attribute of the columns' dataset.
    If split adjustments are to be applied, a loader, split-adjusted columns,
    and the split-adjusted asof-date must be supplied.

    Parameters
    ----------
    estimates : pd.DataFrame
        The raw estimates data.
        ``estimates`` must contain at least 5 columns:
            sid : int64
                The asset id associated with each estimate.

            event_date : datetime64[ns]
                The date on which the event that the estimate is for will/has
                occurred..

            timestamp : datetime64[ns]
                The datetime where we learned about the estimate.

            fiscal_quarter : int64
                The quarter during which the event has/will occur.

            fiscal_year : int64
                The year during which the event has/will occur.

    name_map : dict[str -> str]
        A map of names of BoundColumns that this loader will load to the
        names of the corresponding columns in `events`.
    """

    def __init__(self, estimates, name_map):
        if False:
            print('Hello World!')
        validate_column_specs(estimates, name_map)
        self.estimates = estimates[estimates[EVENT_DATE_FIELD_NAME].notnull() & estimates[FISCAL_QUARTER_FIELD_NAME].notnull() & estimates[FISCAL_YEAR_FIELD_NAME].notnull()]
        self.estimates[NORMALIZED_QUARTERS] = normalize_quarters(self.estimates[FISCAL_YEAR_FIELD_NAME], self.estimates[FISCAL_QUARTER_FIELD_NAME])
        self.array_overwrites_dict = {datetime64ns_dtype: Datetime641DArrayOverwrite, float64_dtype: Float641DArrayOverwrite}
        self.scalar_overwrites_dict = {datetime64ns_dtype: Datetime64Overwrite, float64_dtype: Float64Overwrite}
        self.name_map = name_map

    @abstractmethod
    def get_zeroth_quarter_idx(self, stacked_last_per_qtr):
        if False:
            return 10
        raise NotImplementedError('get_zeroth_quarter_idx')

    @abstractmethod
    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        if False:
            return 10
        raise NotImplementedError('get_shifted_qtrs')

    @abstractmethod
    def create_overwrite_for_estimate(self, column, column_name, last_per_qtr, next_qtr_start_idx, requested_quarter, sid, sid_idx, col_to_split_adjustments, split_adjusted_asof_idx):
        if False:
            print('Hello World!')
        raise NotImplementedError('create_overwrite_for_estimate')

    @abstractproperty
    def searchsorted_side(self):
        if False:
            while True:
                i = 10
        return NotImplementedError('searchsorted_side')

    def get_requested_quarter_data(self, zero_qtr_data, zeroth_quarter_idx, stacked_last_per_qtr, num_announcements, dates):
        if False:
            i = 10
            return i + 15
        "\n        Selects the requested data for each date.\n\n        Parameters\n        ----------\n        zero_qtr_data : pd.DataFrame\n            The 'time zero' data for each calendar date per sid.\n        zeroth_quarter_idx : pd.Index\n            An index of calendar dates, sid, and normalized quarters, for only\n            the rows that have a next or previous earnings estimate.\n        stacked_last_per_qtr : pd.DataFrame\n            The latest estimate known with the dates, normalized quarter, and\n            sid as the index.\n        num_announcements : int\n            The number of annoucements out the user requested relative to\n            each date in the calendar dates.\n        dates : pd.DatetimeIndex\n            The calendar dates for which estimates data is requested.\n\n        Returns\n        --------\n        requested_qtr_data : pd.DataFrame\n            The DataFrame with the latest values for the requested quarter\n            for all columns; `dates` are the index and columns are a MultiIndex\n            with sids at the top level and the dataset columns on the bottom.\n        "
        zero_qtr_data_idx = zero_qtr_data.index
        requested_qtr_idx = pd.MultiIndex.from_arrays([zero_qtr_data_idx.get_level_values(0), zero_qtr_data_idx.get_level_values(1), self.get_shifted_qtrs(zeroth_quarter_idx.get_level_values(NORMALIZED_QUARTERS), num_announcements)], names=[zero_qtr_data_idx.names[0], zero_qtr_data_idx.names[1], SHIFTED_NORMALIZED_QTRS])
        requested_qtr_data = stacked_last_per_qtr.loc[requested_qtr_idx]
        requested_qtr_data = requested_qtr_data.reset_index(SHIFTED_NORMALIZED_QTRS)
        (requested_qtr_data[FISCAL_YEAR_FIELD_NAME], requested_qtr_data[FISCAL_QUARTER_FIELD_NAME]) = split_normalized_quarters(requested_qtr_data[SHIFTED_NORMALIZED_QTRS])
        return requested_qtr_data.unstack(SID_FIELD_NAME).reindex(dates)

    def get_split_adjusted_asof_idx(self, dates):
        if False:
            while True:
                i = 10
        '\n        Compute the index in `dates` where the split-adjusted-asof-date\n        falls. This is the date up to which, and including which, we will\n        need to unapply all adjustments for and then re-apply them as they\n        come in. After this date, adjustments are applied as normal.\n\n        Parameters\n        ----------\n        dates : pd.DatetimeIndex\n            The calendar dates over which the Pipeline is being computed.\n\n        Returns\n        -------\n        split_adjusted_asof_idx : int\n            The index in `dates` at which the data should be split.\n        '
        split_adjusted_asof_idx = dates.searchsorted(self._split_adjusted_asof)
        if split_adjusted_asof_idx == len(dates):
            split_adjusted_asof_idx = len(dates) - 1
        elif self._split_adjusted_asof < dates[0].tz_localize(None):
            split_adjusted_asof_idx = -1
        return split_adjusted_asof_idx

    def collect_overwrites_for_sid(self, group, dates, requested_qtr_data, last_per_qtr, sid_idx, columns, all_adjustments_for_sid, sid):
        if False:
            while True:
                i = 10
        "\n        Given a sid, collect all overwrites that should be applied for this\n        sid at each quarter boundary.\n\n        Parameters\n        ----------\n        group : pd.DataFrame\n            The data for `sid`.\n        dates : pd.DatetimeIndex\n            The calendar dates for which estimates data is requested.\n        requested_qtr_data : pd.DataFrame\n            The DataFrame with the latest values for the requested quarter\n            for all columns.\n        last_per_qtr : pd.DataFrame\n            A DataFrame with a column MultiIndex of [self.estimates.columns,\n            normalized_quarters, sid] that allows easily getting the timeline\n            of estimates for a particular sid for a particular quarter.\n        sid_idx : int\n            The sid's index in the asset index.\n        columns : list of BoundColumn\n            The columns for which the overwrites should be computed.\n        all_adjustments_for_sid : dict[int -> AdjustedArray]\n            A dictionary of the integer index of each timestamp into the date\n            index, mapped to adjustments that should be applied at that\n            index for the given sid (`sid`). This dictionary is modified as\n            adjustments are collected.\n        sid : int\n            The sid for which overwrites should be computed.\n        "
        if len(dates) == 1:
            return
        next_qtr_start_indices = dates.searchsorted(group[EVENT_DATE_FIELD_NAME].values, side=self.searchsorted_side)
        qtrs_with_estimates = group.index.get_level_values(NORMALIZED_QUARTERS).values
        for idx in next_qtr_start_indices:
            if 0 < idx < len(dates):
                requested_quarter = requested_qtr_data[SHIFTED_NORMALIZED_QTRS, sid].iloc[idx]
                self.create_overwrites_for_quarter(all_adjustments_for_sid, idx, last_per_qtr, qtrs_with_estimates, requested_quarter, sid, sid_idx, columns)

    def get_adjustments_for_sid(self, group, dates, requested_qtr_data, last_per_qtr, sid_to_idx, columns, col_to_all_adjustments, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n\n        Parameters\n        ----------\n        group : pd.DataFrame\n            The data for the given sid.\n        dates : pd.DatetimeIndex\n            The calendar dates for which estimates data is requested.\n        requested_qtr_data : pd.DataFrame\n            The DataFrame with the latest values for the requested quarter\n            for all columns.\n        last_per_qtr : pd.DataFrame\n            A DataFrame with a column MultiIndex of [self.estimates.columns,\n            normalized_quarters, sid] that allows easily getting the timeline\n            of estimates for a particular sid for a particular quarter.\n        sid_to_idx : dict[int -> int]\n            A dictionary mapping sid to he sid's index in the asset index.\n        columns : list of BoundColumn\n            The columns for which the overwrites should be computed.\n        col_to_all_adjustments : dict[int -> AdjustedArray]\n            A dictionary of the integer index of each timestamp into the date\n            index, mapped to adjustments that should be applied at that\n            index. This dictionary is for adjustments for ALL sids. It is\n            modified as adjustments are collected.\n        kwargs :\n            Additional arguments used in collecting adjustments; unused here.\n        "
        all_adjustments_for_sid = {}
        sid = int(group.name)
        self.collect_overwrites_for_sid(group, dates, requested_qtr_data, last_per_qtr, sid_to_idx[sid], columns, all_adjustments_for_sid, sid)
        self.merge_into_adjustments_for_all_sids(all_adjustments_for_sid, col_to_all_adjustments)

    def merge_into_adjustments_for_all_sids(self, all_adjustments_for_sid, col_to_all_adjustments):
        if False:
            print('Hello World!')
        '\n        Merge adjustments for a particular sid into a dictionary containing\n        adjustments for all sids.\n\n        Parameters\n        ----------\n        all_adjustments_for_sid : dict[int -> AdjustedArray]\n            All adjustments for a particular sid.\n        col_to_all_adjustments : dict[int -> AdjustedArray]\n            All adjustments for all sids.\n        '
        for col_name in all_adjustments_for_sid:
            if col_name not in col_to_all_adjustments:
                col_to_all_adjustments[col_name] = {}
            for ts in all_adjustments_for_sid[col_name]:
                adjs = all_adjustments_for_sid[col_name][ts]
                add_new_adjustments(col_to_all_adjustments, adjs, col_name, ts)

    def get_adjustments(self, zero_qtr_data, requested_qtr_data, last_per_qtr, dates, assets, columns, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates an AdjustedArray from the given estimates data for the given\n        dates.\n\n        Parameters\n        ----------\n        zero_qtr_data : pd.DataFrame\n            The 'time zero' data for each calendar date per sid.\n        requested_qtr_data : pd.DataFrame\n            The requested quarter data for each calendar date per sid.\n        last_per_qtr : pd.DataFrame\n            A DataFrame with a column MultiIndex of [self.estimates.columns,\n            normalized_quarters, sid] that allows easily getting the timeline\n            of estimates for a particular sid for a particular quarter.\n        dates : pd.DatetimeIndex\n            The calendar dates for which estimates data is requested.\n        assets : pd.Int64Index\n            An index of all the assets from the raw data.\n        columns : list of BoundColumn\n            The columns for which adjustments need to be calculated.\n        kwargs :\n            Additional keyword arguments that should be forwarded to\n            `get_adjustments_for_sid` and to be used in computing adjustments\n            for each sid.\n\n        Returns\n        -------\n        col_to_all_adjustments : dict[int -> AdjustedArray]\n            A dictionary of all adjustments that should be applied.\n        "
        zero_qtr_data.sort_index(inplace=True)
        quarter_shifts = zero_qtr_data.groupby(level=[SID_FIELD_NAME, NORMALIZED_QUARTERS]).nth(-1)
        col_to_all_adjustments = {}
        sid_to_idx = dict(zip(assets, range(len(assets))))
        quarter_shifts.groupby(level=SID_FIELD_NAME).apply(self.get_adjustments_for_sid, dates, requested_qtr_data, last_per_qtr, sid_to_idx, columns, col_to_all_adjustments, **kwargs)
        return col_to_all_adjustments

    def create_overwrites_for_quarter(self, col_to_overwrites, next_qtr_start_idx, last_per_qtr, quarters_with_estimates_for_sid, requested_quarter, sid, sid_idx, columns):
        if False:
            while True:
                i = 10
        "\n        Add entries to the dictionary of columns to adjustments for the given\n        sid and the given quarter.\n\n        Parameters\n        ----------\n        col_to_overwrites : dict [column_name -> list of ArrayAdjustment]\n            A dictionary mapping column names to all overwrites for those\n            columns.\n        next_qtr_start_idx : int\n            The index of the first day of the next quarter in the calendar\n            dates.\n        last_per_qtr : pd.DataFrame\n            A DataFrame with a column MultiIndex of [self.estimates.columns,\n            normalized_quarters, sid] that allows easily getting the timeline\n            of estimates for a particular sid for a particular quarter; this\n            is particularly useful for getting adjustments for 'next'\n            estimates.\n        quarters_with_estimates_for_sid : np.array\n            An array of all quarters for which there are estimates for the\n            given sid.\n        requested_quarter : float\n            The quarter for which the overwrite should be created.\n        sid : int\n            The sid for which to create overwrites.\n        sid_idx : int\n            The index of the sid in `assets`.\n        columns : list of BoundColumn\n            The columns for which to create overwrites.\n        "
        for col in columns:
            column_name = self.name_map[col.name]
            if column_name not in col_to_overwrites:
                col_to_overwrites[column_name] = {}
            if requested_quarter in quarters_with_estimates_for_sid:
                adjs = self.create_overwrite_for_estimate(col, column_name, last_per_qtr, next_qtr_start_idx, requested_quarter, sid, sid_idx)
                add_new_adjustments(col_to_overwrites, adjs, column_name, next_qtr_start_idx)
            else:
                adjs = [self.overwrite_with_null(col, next_qtr_start_idx, sid_idx)]
                add_new_adjustments(col_to_overwrites, adjs, column_name, next_qtr_start_idx)

    def overwrite_with_null(self, column, next_qtr_start_idx, sid_idx):
        if False:
            return 10
        return self.scalar_overwrites_dict[column.dtype](0, next_qtr_start_idx - 1, sid_idx, sid_idx, column.missing_value)

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        if False:
            return 10
        col_to_datasets = {col: col.dataset for col in columns}
        try:
            groups = groupby(lambda col: col_to_datasets[col].num_announcements, col_to_datasets)
        except AttributeError:
            raise AttributeError('Datasets loaded via the EarningsEstimatesLoader must define a `num_announcements` attribute that defines how many quarters out the loader should load the data relative to `dates`.')
        if any((num_qtr < 0 for num_qtr in groups)):
            raise ValueError(INVALID_NUM_QTRS_MESSAGE % ','.join((str(qtr) for qtr in groups if qtr < 0)))
        out = {}
        data_query_cutoff_times = domain.data_query_cutoff_for_sessions(dates)
        assets_with_data = set(sids) & set(self.estimates[SID_FIELD_NAME])
        (last_per_qtr, stacked_last_per_qtr) = self.get_last_data_per_qtr(assets_with_data, columns, dates, data_query_cutoff_times)
        zeroth_quarter_idx = self.get_zeroth_quarter_idx(stacked_last_per_qtr)
        zero_qtr_data = stacked_last_per_qtr.loc[zeroth_quarter_idx]
        for (num_announcements, columns) in groups.items():
            requested_qtr_data = self.get_requested_quarter_data(zero_qtr_data, zeroth_quarter_idx, stacked_last_per_qtr, num_announcements, dates)
            col_to_adjustments = self.get_adjustments(zero_qtr_data, requested_qtr_data, last_per_qtr, dates, sids, columns)
            asset_indexer = sids.get_indexer_for(requested_qtr_data.columns.levels[1])
            for col in columns:
                column_name = self.name_map[col.name]
                output_array = np.full((len(dates), len(sids)), col.missing_value, dtype=col.dtype)
                output_array[:, asset_indexer] = requested_qtr_data[column_name].values
                out[col] = AdjustedArray(output_array, dict(col_to_adjustments.get(column_name, {})), col.missing_value)
        return out

    def get_last_data_per_qtr(self, assets_with_data, columns, dates, data_query_cutoff_times):
        if False:
            while True:
                i = 10
        '\n        Determine the last piece of information we know for each column on each\n        date in the index for each sid and quarter.\n\n        Parameters\n        ----------\n        assets_with_data : pd.Index\n            Index of all assets that appear in the raw data given to the\n            loader.\n        columns : iterable of BoundColumn\n            The columns that need to be loaded from the raw data.\n        data_query_cutoff_times : pd.DatetimeIndex\n            The calendar of dates for which data should be loaded.\n\n        Returns\n        -------\n        stacked_last_per_qtr : pd.DataFrame\n            A DataFrame indexed by [dates, sid, normalized_quarters] that has\n            the latest information for each row of the index, sorted by event\n            date.\n        last_per_qtr : pd.DataFrame\n            A DataFrame with columns that are a MultiIndex of [\n            self.estimates.columns, normalized_quarters, sid].\n        '
        last_per_qtr = last_in_date_group(self.estimates, data_query_cutoff_times, assets_with_data, reindex=True, extra_groupers=[NORMALIZED_QUARTERS])
        last_per_qtr.index = dates
        ffill_across_cols(last_per_qtr, columns, self.name_map)
        stacked_last_per_qtr = last_per_qtr.stack([SID_FIELD_NAME, NORMALIZED_QUARTERS])
        stacked_last_per_qtr.index.set_names(SIMULATION_DATES, level=0, inplace=True)
        stacked_last_per_qtr = stacked_last_per_qtr.sort_values(EVENT_DATE_FIELD_NAME)
        stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] = pd.to_datetime(stacked_last_per_qtr[EVENT_DATE_FIELD_NAME])
        return (last_per_qtr, stacked_last_per_qtr)

class NextEarningsEstimatesLoader(EarningsEstimatesLoader):
    searchsorted_side = 'right'

    def create_overwrite_for_estimate(self, column, column_name, last_per_qtr, next_qtr_start_idx, requested_quarter, sid, sid_idx, col_to_split_adjustments=None, split_adjusted_asof_idx=None):
        if False:
            i = 10
            return i + 15
        return [self.array_overwrites_dict[column.dtype](0, next_qtr_start_idx - 1, sid_idx, sid_idx, last_per_qtr[column_name, requested_quarter, sid].values[:next_qtr_start_idx])]

    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        if False:
            i = 10
            return i + 15
        return zero_qtrs + (num_announcements - 1)

    def get_zeroth_quarter_idx(self, stacked_last_per_qtr):
        if False:
            print('Hello World!')
        "\n        Filters for releases that are on or after each simulation date and\n        determines the next quarter by picking out the upcoming release for\n        each date in the index.\n\n        Parameters\n        ----------\n        stacked_last_per_qtr : pd.DataFrame\n            A DataFrame with index of calendar dates, sid, and normalized\n            quarters with each row being the latest estimate for the row's\n            index values, sorted by event date.\n\n        Returns\n        -------\n        next_releases_per_date_index : pd.MultiIndex\n            An index of calendar dates, sid, and normalized quarters, for only\n            the rows that have a next event.\n        "
        next_releases_per_date = stacked_last_per_qtr.loc[stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] >= stacked_last_per_qtr.index.get_level_values(SIMULATION_DATES)].groupby(level=[SIMULATION_DATES, SID_FIELD_NAME], as_index=False).nth(0)
        return next_releases_per_date.index

class PreviousEarningsEstimatesLoader(EarningsEstimatesLoader):
    searchsorted_side = 'left'

    def create_overwrite_for_estimate(self, column, column_name, dates, next_qtr_start_idx, requested_quarter, sid, sid_idx, col_to_split_adjustments=None, split_adjusted_asof_idx=None, split_dict=None):
        if False:
            i = 10
            return i + 15
        return [self.overwrite_with_null(column, next_qtr_start_idx, sid_idx)]

    def get_shifted_qtrs(self, zero_qtrs, num_announcements):
        if False:
            return 10
        return zero_qtrs - (num_announcements - 1)

    def get_zeroth_quarter_idx(self, stacked_last_per_qtr):
        if False:
            print('Hello World!')
        "\n        Filters for releases that are on or after each simulation date and\n        determines the previous quarter by picking out the most recent\n        release relative to each date in the index.\n\n        Parameters\n        ----------\n        stacked_last_per_qtr : pd.DataFrame\n            A DataFrame with index of calendar dates, sid, and normalized\n            quarters with each row being the latest estimate for the row's\n            index values, sorted by event date.\n\n        Returns\n        -------\n        previous_releases_per_date_index : pd.MultiIndex\n            An index of calendar dates, sid, and normalized quarters, for only\n            the rows that have a previous event.\n        "
        previous_releases_per_date = stacked_last_per_qtr.loc[stacked_last_per_qtr[EVENT_DATE_FIELD_NAME] <= stacked_last_per_qtr.index.get_level_values(SIMULATION_DATES)].groupby(level=[SIMULATION_DATES, SID_FIELD_NAME], as_index=False).nth(-1)
        return previous_releases_per_date.index

def validate_split_adjusted_column_specs(name_map, columns):
    if False:
        i = 10
        return i + 15
    to_be_split = set(columns)
    available = set(name_map.keys())
    extra = to_be_split - available
    if extra:
        raise ValueError('EarningsEstimatesLoader got the following extra columns to be split-adjusted: {extra}.\nGot Columns: {to_be_split}\nAvailable Columns: {available}'.format(extra=sorted(extra), to_be_split=sorted(to_be_split), available=sorted(available)))

class SplitAdjustedEstimatesLoader(EarningsEstimatesLoader):
    """
    Estimates loader that loads data that needs to be split-adjusted.

    Parameters
    ----------
    split_adjustments_loader : SQLiteAdjustmentReader
        The loader to use for reading split adjustments.
    split_adjusted_column_names : iterable of str
        The column names that should be split-adjusted.
    split_adjusted_asof : pd.Timestamp
        The date that separates data into 2 halves: the first half is the set
        of dates up to and including the split_adjusted_asof date. All
        adjustments occurring during this first half are applied  to all
        dates in this first half. The second half is the set of dates after
        the split_adjusted_asof date. All adjustments occurring during this
        second half are applied sequentially as they appear in the timeline.
    """

    def __init__(self, estimates, name_map, split_adjustments_loader, split_adjusted_column_names, split_adjusted_asof):
        if False:
            return 10
        validate_split_adjusted_column_specs(name_map, split_adjusted_column_names)
        self._split_adjustments = split_adjustments_loader
        self._split_adjusted_column_names = split_adjusted_column_names
        self._split_adjusted_asof = split_adjusted_asof
        self._split_adjustment_dict = {}
        super(SplitAdjustedEstimatesLoader, self).__init__(estimates, name_map)

    @abstractmethod
    def collect_split_adjustments(self, adjustments_for_sid, requested_qtr_data, dates, sid, sid_idx, sid_estimates, split_adjusted_asof_idx, pre_adjustments, post_adjustments, requested_split_adjusted_columns):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('collect_split_adjustments')

    def get_adjustments_for_sid(self, group, dates, requested_qtr_data, last_per_qtr, sid_to_idx, columns, col_to_all_adjustments, split_adjusted_asof_idx=None, split_adjusted_cols_for_group=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Collects both overwrites and adjustments for a particular sid.\n\n        Parameters\n        ----------\n        split_adjusted_asof_idx : int\n            The integer index of the date on which the data was split-adjusted.\n        split_adjusted_cols_for_group : list of str\n            The names of requested columns that should also be split-adjusted.\n        '
        all_adjustments_for_sid = {}
        sid = int(group.name)
        self.collect_overwrites_for_sid(group, dates, requested_qtr_data, last_per_qtr, sid_to_idx[sid], columns, all_adjustments_for_sid, sid)
        (pre_adjustments, post_adjustments) = self.retrieve_split_adjustment_data_for_sid(dates, sid, split_adjusted_asof_idx)
        sid_estimates = self.estimates[self.estimates[SID_FIELD_NAME] == sid]
        for col_name in split_adjusted_cols_for_group:
            if col_name not in all_adjustments_for_sid:
                all_adjustments_for_sid[col_name] = {}
        self.collect_split_adjustments(all_adjustments_for_sid, requested_qtr_data, dates, sid, sid_to_idx[sid], sid_estimates, split_adjusted_asof_idx, pre_adjustments, post_adjustments, split_adjusted_cols_for_group)
        self.merge_into_adjustments_for_all_sids(all_adjustments_for_sid, col_to_all_adjustments)

    def get_adjustments(self, zero_qtr_data, requested_qtr_data, last_per_qtr, dates, assets, columns, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Calculates both split adjustments and overwrites for all sids.\n        '
        split_adjusted_cols_for_group = [self.name_map[col.name] for col in columns if self.name_map[col.name] in self._split_adjusted_column_names]
        split_adjusted_asof_idx = self.get_split_adjusted_asof_idx(dates)
        return super(SplitAdjustedEstimatesLoader, self).get_adjustments(zero_qtr_data, requested_qtr_data, last_per_qtr, dates, assets, columns, split_adjusted_cols_for_group=split_adjusted_cols_for_group, split_adjusted_asof_idx=split_adjusted_asof_idx)

    def determine_end_idx_for_adjustment(self, adjustment_ts, dates, upper_bound, requested_quarter, sid_estimates):
        if False:
            i = 10
            return i + 15
        "\n        Determines the date until which the adjustment at the given date\n        index should be applied for the given quarter.\n\n        Parameters\n        ----------\n        adjustment_ts : pd.Timestamp\n            The timestamp at which the adjustment occurs.\n        dates : pd.DatetimeIndex\n            The calendar dates over which the Pipeline is being computed.\n        upper_bound : int\n            The index of the upper bound in the calendar dates. This is the\n            index until which the adjusment will be applied unless there is\n            information for the requested quarter that comes in on or before\n            that date.\n        requested_quarter : float\n            The quarter for which we are determining how the adjustment\n            should be applied.\n        sid_estimates : pd.DataFrame\n            The DataFrame of estimates data for the sid for which we're\n            applying the given adjustment.\n\n        Returns\n        -------\n        end_idx : int\n            The last index to which the adjustment should be applied for the\n            given quarter/sid.\n        "
        end_idx = upper_bound
        newest_kd_for_qtr = sid_estimates[(sid_estimates[NORMALIZED_QUARTERS] == requested_quarter) & (sid_estimates[TS_FIELD_NAME] >= adjustment_ts)][TS_FIELD_NAME].min()
        if pd.notnull(newest_kd_for_qtr):
            newest_kd_idx = dates.searchsorted(newest_kd_for_qtr)
            if newest_kd_idx <= upper_bound:
                end_idx = newest_kd_idx - 1
        return end_idx

    def collect_pre_split_asof_date_adjustments(self, split_adjusted_asof_date_idx, sid_idx, pre_adjustments, requested_split_adjusted_columns):
        if False:
            print('Hello World!')
        '\n        Collect split adjustments that occur before the\n        split-adjusted-asof-date. All those adjustments must first be\n        UN-applied at the first date index and then re-applied on the\n        appropriate dates in order to match point in time share pricing data.\n\n        Parameters\n        ----------\n        split_adjusted_asof_date_idx : int\n            The index in the calendar dates as-of which all data was\n            split-adjusted.\n        sid_idx : int\n            The index of the sid for which adjustments should be collected in\n            the adjusted array.\n        pre_adjustments : tuple(list(float), list(int))\n            The adjustment values, indexes in `dates`, and timestamps for\n            adjustments that happened after the split-asof-date.\n        requested_split_adjusted_columns : list of str\n            The requested split adjusted columns.\n\n        Returns\n        -------\n        col_to_split_adjustments : dict[str -> dict[int -> list of Adjustment]]\n            The adjustments for this sid that occurred on or before the\n            split-asof-date.\n        '
        col_to_split_adjustments = {}
        if len(pre_adjustments[0]):
            (adjustment_values, date_indexes) = pre_adjustments
            for column_name in requested_split_adjusted_columns:
                col_to_split_adjustments[column_name] = {}
                col_to_split_adjustments[column_name][0] = [Float64Multiply(0, split_adjusted_asof_date_idx, sid_idx, sid_idx, 1 / future_adjustment) for future_adjustment in adjustment_values]
                for (adjustment, date_index) in zip(adjustment_values, date_indexes):
                    adj = Float64Multiply(0, split_adjusted_asof_date_idx, sid_idx, sid_idx, adjustment)
                    add_new_adjustments(col_to_split_adjustments, [adj], column_name, date_index)
        return col_to_split_adjustments

    def collect_post_asof_split_adjustments(self, post_adjustments, requested_qtr_data, sid, sid_idx, sid_estimates, requested_split_adjusted_columns):
        if False:
            i = 10
            return i + 15
        '\n        Collect split adjustments that occur after the\n        split-adjusted-asof-date. Each adjustment needs to be applied to all\n        dates on which knowledge for the requested quarter was older than the\n        date of the adjustment.\n\n        Parameters\n        ----------\n        post_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)\n            The adjustment values, indexes in `dates`, and timestamps for\n            adjustments that happened after the split-asof-date.\n        requested_qtr_data : pd.DataFrame\n            The requested quarter data for each calendar date per sid.\n        sid : int\n            The sid for which adjustments need to be collected.\n        sid_idx : int\n            The index of `sid` in the adjusted array.\n        sid_estimates : pd.DataFrame\n            The raw estimates data for this sid.\n        requested_split_adjusted_columns : list of str\n            The requested split adjusted columns.\n        Returns\n        -------\n        col_to_split_adjustments : dict[str -> dict[int -> list of Adjustment]]\n            The adjustments for this sid that occurred after the\n            split-asof-date.\n        '
        col_to_split_adjustments = {}
        if post_adjustments:
            requested_qtr_timeline = requested_qtr_data[SHIFTED_NORMALIZED_QTRS][sid].reset_index()
            requested_qtr_timeline = requested_qtr_timeline[requested_qtr_timeline[sid].notnull()]
            qtr_ranges_idxs = np.split(requested_qtr_timeline.index, np.where(np.diff(requested_qtr_timeline[sid]) != 0)[0] + 1)
            requested_quarters_per_range = [requested_qtr_timeline[sid][r[0]] for r in qtr_ranges_idxs]
            for (i, qtr_range) in enumerate(qtr_ranges_idxs):
                for (adjustment, date_index, timestamp) in zip(*post_adjustments):
                    upper_bound = qtr_range[-1]
                    end_idx = self.determine_end_idx_for_adjustment(timestamp, requested_qtr_data.index, upper_bound, requested_quarters_per_range[i], sid_estimates)
                    start_idx = qtr_range[0]
                    if date_index > start_idx:
                        start_idx = date_index
                    if qtr_range[0] <= end_idx:
                        for column_name in requested_split_adjusted_columns:
                            if column_name not in col_to_split_adjustments:
                                col_to_split_adjustments[column_name] = {}
                            adj = Float64Multiply(qtr_range[0], end_idx, sid_idx, sid_idx, adjustment)
                            add_new_adjustments(col_to_split_adjustments, [adj], column_name, start_idx)
        return col_to_split_adjustments

    def retrieve_split_adjustment_data_for_sid(self, dates, sid, split_adjusted_asof_idx):
        if False:
            for i in range(10):
                print('nop')
        '\n        dates : pd.DatetimeIndex\n            The calendar dates.\n        sid : int\n            The sid for which we want to retrieve adjustments.\n        split_adjusted_asof_idx : int\n            The index in `dates` as-of which the data is split adjusted.\n\n        Returns\n        -------\n        pre_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)\n            The adjustment values and indexes in `dates` for\n            adjustments that happened before the split-asof-date.\n        post_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)\n            The adjustment values, indexes in `dates`, and timestamps for\n            adjustments that happened after the split-asof-date.\n        '
        adjustments = self._split_adjustments.get_adjustments_for_sid('splits', sid)
        sorted(adjustments, key=lambda adj: adj[0])
        adjustments = list(filter(lambda x: dates[0] <= x[0] <= dates[-1], adjustments))
        adjustment_values = np.array([adj[1] for adj in adjustments])
        timestamps = pd.DatetimeIndex([adj[0] for adj in adjustments])
        date_indexes = dates.searchsorted(timestamps)
        pre_adjustment_idxs = np.where(date_indexes <= split_adjusted_asof_idx)[0]
        last_adjustment_split_asof_idx = -1
        if len(pre_adjustment_idxs):
            last_adjustment_split_asof_idx = pre_adjustment_idxs.max()
        pre_adjustments = (adjustment_values[:last_adjustment_split_asof_idx + 1], date_indexes[:last_adjustment_split_asof_idx + 1])
        post_adjustments = (adjustment_values[last_adjustment_split_asof_idx + 1:], date_indexes[last_adjustment_split_asof_idx + 1:], timestamps[last_adjustment_split_asof_idx + 1:])
        return (pre_adjustments, post_adjustments)

    def _collect_adjustments(self, requested_qtr_data, sid, sid_idx, sid_estimates, split_adjusted_asof_idx, pre_adjustments, post_adjustments, requested_split_adjusted_columns):
        if False:
            while True:
                i = 10
        pre_adjustments_dict = self.collect_pre_split_asof_date_adjustments(split_adjusted_asof_idx, sid_idx, pre_adjustments, requested_split_adjusted_columns)
        post_adjustments_dict = self.collect_post_asof_split_adjustments(post_adjustments, requested_qtr_data, sid, sid_idx, sid_estimates, requested_split_adjusted_columns)
        return (pre_adjustments_dict, post_adjustments_dict)

    def merge_split_adjustments_with_overwrites(self, pre, post, overwrites, requested_split_adjusted_columns):
        if False:
            print('Hello World!')
        '\n        Merge split adjustments with the dict containing overwrites.\n\n        Parameters\n        ----------\n        pre : dict[str -> dict[int -> list]]\n            The adjustments that occur before the split-adjusted-asof-date.\n        post : dict[str -> dict[int -> list]]\n            The adjustments that occur after the split-adjusted-asof-date.\n        overwrites : dict[str -> dict[int -> list]]\n            The overwrites across all time. Adjustments will be merged into\n            this dictionary.\n        requested_split_adjusted_columns : list of str\n            List of names of split adjusted columns that are being requested.\n        '
        for column_name in requested_split_adjusted_columns:
            if pre:
                for ts in pre[column_name]:
                    add_new_adjustments(overwrites, pre[column_name][ts], column_name, ts)
            if post:
                for ts in post[column_name]:
                    add_new_adjustments(overwrites, post[column_name][ts], column_name, ts)

class PreviousSplitAdjustedEarningsEstimatesLoader(SplitAdjustedEstimatesLoader, PreviousEarningsEstimatesLoader):

    def collect_split_adjustments(self, adjustments_for_sid, requested_qtr_data, dates, sid, sid_idx, sid_estimates, split_adjusted_asof_idx, pre_adjustments, post_adjustments, requested_split_adjusted_columns):
        if False:
            return 10
        "\n        Collect split adjustments for previous quarters and apply them to the\n        given dictionary of splits for the given sid. Since overwrites just\n        replace all estimates before the new quarter with NaN, we don't need to\n        worry about re-applying split adjustments.\n\n        Parameters\n        ----------\n        adjustments_for_sid : dict[str -> dict[int -> list]]\n            The dictionary of adjustments to which splits need to be added.\n            Initially it contains only overwrites.\n        requested_qtr_data : pd.DataFrame\n            The requested quarter data for each calendar date per sid.\n        dates : pd.DatetimeIndex\n            The calendar dates for which estimates data is requested.\n        sid : int\n            The sid for which adjustments need to be collected.\n        sid_idx : int\n            The index of `sid` in the adjusted array.\n        sid_estimates : pd.DataFrame\n            The raw estimates data for the given sid.\n        split_adjusted_asof_idx : int\n            The index in `dates` as-of which the data is split adjusted.\n        pre_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)\n            The adjustment values and indexes in `dates` for\n            adjustments that happened before the split-asof-date.\n        post_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)\n            The adjustment values, indexes in `dates`, and timestamps for\n            adjustments that happened after the split-asof-date.\n        requested_split_adjusted_columns : list of str\n            List of requested split adjusted column names.\n        "
        (pre_adjustments_dict, post_adjustments_dict) = self._collect_adjustments(requested_qtr_data, sid, sid_idx, sid_estimates, split_adjusted_asof_idx, pre_adjustments, post_adjustments, requested_split_adjusted_columns)
        self.merge_split_adjustments_with_overwrites(pre_adjustments_dict, post_adjustments_dict, adjustments_for_sid, requested_split_adjusted_columns)

class NextSplitAdjustedEarningsEstimatesLoader(SplitAdjustedEstimatesLoader, NextEarningsEstimatesLoader):

    def collect_split_adjustments(self, adjustments_for_sid, requested_qtr_data, dates, sid, sid_idx, sid_estimates, split_adjusted_asof_idx, pre_adjustments, post_adjustments, requested_split_adjusted_columns):
        if False:
            for i in range(10):
                print('nop')
        '\n        Collect split adjustments for future quarters. Re-apply adjustments\n        that would be overwritten by overwrites. Merge split adjustments with\n        overwrites into the given dictionary of splits for the given sid.\n\n        Parameters\n        ----------\n        adjustments_for_sid : dict[str -> dict[int -> list]]\n            The dictionary of adjustments to which splits need to be added.\n            Initially it contains only overwrites.\n        requested_qtr_data : pd.DataFrame\n            The requested quarter data for each calendar date per sid.\n        dates : pd.DatetimeIndex\n            The calendar dates for which estimates data is requested.\n        sid : int\n            The sid for which adjustments need to be collected.\n        sid_idx : int\n            The index of `sid` in the adjusted array.\n        sid_estimates : pd.DataFrame\n            The raw estimates data for the given sid.\n        split_adjusted_asof_idx : int\n            The index in `dates` as-of which the data is split adjusted.\n        pre_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)\n            The adjustment values and indexes in `dates` for\n            adjustments that happened before the split-asof-date.\n        post_adjustments : tuple(list(float), list(int), pd.DatetimeIndex)\n            The adjustment values, indexes in `dates`, and timestamps for\n            adjustments that happened after the split-asof-date.\n        requested_split_adjusted_columns : list of str\n            List of requested split adjusted column names.\n        '
        (pre_adjustments_dict, post_adjustments_dict) = self._collect_adjustments(requested_qtr_data, sid, sid_idx, sid_estimates, split_adjusted_asof_idx, pre_adjustments, post_adjustments, requested_split_adjusted_columns)
        for column_name in requested_split_adjusted_columns:
            for overwrite_ts in adjustments_for_sid[column_name]:
                if overwrite_ts <= split_adjusted_asof_idx and pre_adjustments_dict:
                    for split_ts in pre_adjustments_dict[column_name]:
                        if split_ts < overwrite_ts:
                            adjustments_for_sid[column_name][overwrite_ts].extend([Float64Multiply(0, overwrite_ts - 1, sid_idx, sid_idx, adjustment.value) for adjustment in pre_adjustments_dict[column_name][split_ts]])
                else:
                    requested_quarter = requested_qtr_data[SHIFTED_NORMALIZED_QTRS, sid].iloc[overwrite_ts]
                    for (adjustment_value, date_index, timestamp) in zip(*post_adjustments):
                        if split_adjusted_asof_idx < date_index < overwrite_ts:
                            upper_bound = overwrite_ts - 1
                            end_idx = self.determine_end_idx_for_adjustment(timestamp, dates, upper_bound, requested_quarter, sid_estimates)
                            adjustments_for_sid[column_name][overwrite_ts].append(Float64Multiply(0, end_idx, sid_idx, sid_idx, adjustment_value))
        self.merge_split_adjustments_with_overwrites(pre_adjustments_dict, post_adjustments_dict, adjustments_for_sid, requested_split_adjusted_columns)