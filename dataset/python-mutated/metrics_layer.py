from __future__ import annotations
from typing import Callable, Mapping, Optional, Union
import sentry_sdk
from snuba_sdk import AliasedExpression, Column, Condition, Function, Op
from sentry.api.event_search import SearchFilter
from sentry.exceptions import IncompatibleMetricsQuery, InvalidSearchQuery
from sentry.search.events import builder, constants, fields
from sentry.search.events.datasets import field_aliases, filter_aliases, function_aliases
from sentry.search.events.datasets.metrics import MetricsDatasetConfig
from sentry.search.events.types import SelectType, WhereType
from sentry.snuba.metrics.naming_layer.mri import SessionMRI, TransactionMRI
from sentry.utils.numbers import format_grouped_length

class MetricsLayerDatasetConfig(MetricsDatasetConfig):

    def __init__(self, builder: builder.MetricsQueryBuilder):
        if False:
            for i in range(10):
                print('nop')
        self.builder = builder

    @property
    def search_filter_converter(self) -> Mapping[str, Callable[[SearchFilter], Optional[WhereType]]]:
        if False:
            while True:
                i = 10
        return {constants.PROJECT_ALIAS: self._project_slug_filter_converter, constants.PROJECT_NAME_ALIAS: self._project_slug_filter_converter, constants.EVENT_TYPE_ALIAS: self._event_type_converter, constants.TEAM_KEY_TRANSACTION_ALIAS: self._key_transaction_filter_converter, 'transaction': self._transaction_filter_converter, 'tags[transaction]': self._transaction_filter_converter, constants.TITLE_ALIAS: self._transaction_filter_converter}

    @property
    def field_alias_converter(self) -> Mapping[str, Callable[[str], SelectType]]:
        if False:
            print('Hello World!')
        return {constants.PROJECT_ALIAS: self._resolve_project_slug_alias, constants.PROJECT_NAME_ALIAS: self._resolve_project_slug_alias, constants.TEAM_KEY_TRANSACTION_ALIAS: self._resolve_team_key_transaction_alias, constants.TITLE_ALIAS: self._resolve_title_alias, constants.PROJECT_DOT_ID_ALIAS: lambda alias: AliasedExpression(self.builder.resolve_column(constants.PROJECT_ID_ALIAS), alias)}

    def resolve_mri(self, value: str) -> Column:
        if False:
            return 10
        'Resolve to the MRI'
        metric_mri = constants.METRICS_MAP.get(value)
        if metric_mri is None:
            for measurement in self.builder.custom_measurement_map:
                if measurement['name'] == value and measurement['metric_id'] is not None:
                    return Column(measurement['mri'])
        if metric_mri is None:
            metric_mri = value
        return Column(metric_mri)

    @property
    def function_converter(self) -> Mapping[str, fields.MetricsFunction]:
        if False:
            i = 10
            return i + 15
        "Make sure to update METRIC_FUNCTION_LIST_BY_TYPE when adding functions here, can't be a dynamic list since\n        the Metric Layer will actually handle which dataset each function goes to\n        "
        function_converter = {function.name: function for function in [fields.MetricsFunction('apdex', optional_args=[fields.NullableNumberRange('satisfaction', 0, None)], snql_metric_layer=self._resolve_apdex_function, default_result_type='number'), fields.MetricsFunction('avg', required_args=[fields.MetricArg('column', allowed_columns=constants.METRIC_DURATION_COLUMNS)], snql_metric_layer=lambda args, alias: Function('avg', [self.resolve_mri(args['column'])], alias), default_result_type='integer'), fields.MetricsFunction('count_miserable', required_args=[fields.MetricArg('column', allowed_columns=['user'], allow_custom_measurements=False)], optional_args=[fields.NullableNumberRange('satisfaction', 0, None)], snql_metric_layer=self._resolve_count_miserable_function, default_result_type='integer'), fields.MetricsFunction('user_misery', optional_args=[fields.NullableNumberRange('satisfaction', 0, None)], snql_metric_layer=self._resolve_user_misery_function, default_result_type='number'), fields.MetricsFunction('p50', optional_args=[fields.with_default('transaction.duration', fields.MetricArg('column', allowed_columns=constants.METRIC_DURATION_COLUMNS))], snql_metric_layer=lambda args, alias: function_aliases.resolve_metrics_layer_percentile(args=args, alias=alias, resolve_mri=self.resolve_mri, fixed_percentile=0.5), is_percentile=True, result_type_fn=self.reflective_result_type(), default_result_type='duration'), fields.MetricsFunction('p75', optional_args=[fields.with_default('transaction.duration', fields.MetricArg('column', allowed_columns=constants.METRIC_DURATION_COLUMNS))], snql_metric_layer=lambda args, alias: function_aliases.resolve_metrics_layer_percentile(args=args, alias=alias, resolve_mri=self.resolve_mri, fixed_percentile=0.75), is_percentile=True, result_type_fn=self.reflective_result_type(), default_result_type='duration'), fields.MetricsFunction('p90', optional_args=[fields.with_default('transaction.duration', fields.MetricArg('column', allowed_columns=constants.METRIC_DURATION_COLUMNS))], snql_metric_layer=lambda args, alias: function_aliases.resolve_metrics_layer_percentile(args=args, alias=alias, resolve_mri=self.resolve_mri, fixed_percentile=0.9), is_percentile=True, result_type_fn=self.reflective_result_type(), default_result_type='duration'), fields.MetricsFunction('p95', optional_args=[fields.with_default('transaction.duration', fields.MetricArg('column', allowed_columns=constants.METRIC_DURATION_COLUMNS))], snql_metric_layer=lambda args, alias: function_aliases.resolve_metrics_layer_percentile(args=args, alias=alias, resolve_mri=self.resolve_mri, fixed_percentile=0.95), is_percentile=True, result_type_fn=self.reflective_result_type(), default_result_type='duration'), fields.MetricsFunction('p99', optional_args=[fields.with_default('transaction.duration', fields.MetricArg('column', allowed_columns=constants.METRIC_DURATION_COLUMNS))], snql_metric_layer=lambda args, alias: function_aliases.resolve_metrics_layer_percentile(args=args, alias=alias, resolve_mri=self.resolve_mri, fixed_percentile=0.99), is_percentile=True, result_type_fn=self.reflective_result_type(), default_result_type='duration'), fields.MetricsFunction('p100', optional_args=[fields.with_default('transaction.duration', fields.MetricArg('column', allowed_columns=constants.METRIC_DURATION_COLUMNS))], snql_metric_layer=lambda args, alias: function_aliases.resolve_metrics_layer_percentile(args=args, alias=alias, resolve_mri=self.resolve_mri, fixed_percentile=1), result_type_fn=self.reflective_result_type(), default_result_type='duration'), fields.MetricsFunction('max', required_args=[fields.MetricArg('column')], snql_metric_layer=lambda args, alias: Function('max', [self.resolve_mri(args['column'])], alias), result_type_fn=self.reflective_result_type()), fields.MetricsFunction('min', required_args=[fields.MetricArg('column')], snql_metric_layer=lambda args, alias: Function('min', [self.resolve_mri(args['column'])], alias), result_type_fn=self.reflective_result_type()), fields.MetricsFunction('last', required_args=[fields.MetricArg('column')], snql_metric_layer=lambda args, alias: Function('last', [self.resolve_mri(args['column'])], alias), result_type_fn=self.reflective_result_type()), fields.MetricsFunction('sum', required_args=[fields.MetricArg('column')], snql_metric_layer=lambda args, alias: Function('sum', [self.resolve_mri(args['column'])], alias), result_type_fn=self.reflective_result_type()), fields.MetricsFunction('sumIf', required_args=[fields.MetricArg('if_col', allowed_columns=['session.status']), fields.SnQLStringArg('if_val', allowed_strings=['init', 'crashed'])], snql_metric_layer=lambda args, alias: Function('sum_if_column', [Column(SessionMRI.RAW_SESSION.value), args['if_col'], args['if_val']], alias), default_result_type='integer'), fields.MetricsFunction('percentile', required_args=[fields.with_default('transaction.duration', fields.MetricArg('column', allowed_columns=constants.METRIC_DURATION_COLUMNS)), fields.NumberRange('percentile', 0, 1)], is_percentile=True, snql_metric_layer=lambda args, alias: function_aliases.resolve_metrics_layer_percentile(args=args, alias=alias, resolve_mri=self.resolve_mri, fixed_percentile=args['percentile']), default_result_type='duration'), fields.MetricsFunction('count_unique', required_args=[fields.MetricArg('column', allowed_columns=['user'], allow_custom_measurements=False)], snql_metric_layer=lambda args, alias: Function('count_unique', [self.resolve_mri(args['column'])], alias), default_result_type='integer'), fields.MetricsFunction('uniq', snql_metric_layer=lambda args, alias: Function('count_unique', [Column(SessionMRI.RAW_USER.value)], alias)), fields.MetricsFunction('uniqIf', required_args=[fields.MetricArg('if_col', allowed_columns=['session.status']), fields.SnQLStringArg('if_val', allowed_strings=['crashed'])], snql_metric_layer=lambda args, alias: Function('uniq_if_column', [Column(SessionMRI.RAW_USER.value), args['if_col'], args['if_val']], alias), default_result_type='integer'), fields.MetricsFunction('count', optional_args=[fields.with_default('transaction.duration', fields.MetricArg('column'))], snql_metric_layer=lambda args, alias: Function('count', [self.resolve_mri(args['column'])], alias), default_result_type='integer'), fields.MetricsFunction('count_web_vitals', required_args=[fields.MetricArg('column', allowed_columns=['measurements.fp', 'measurements.fcp', 'measurements.lcp', 'measurements.fid', 'measurements.cls', 'measurements.ttfb'], allow_custom_measurements=False), fields.SnQLStringArg('quality', allowed_strings=['good', 'meh', 'poor', 'any'])], snql_metric_layer=self._resolve_web_vital_function, default_result_type='integer'), fields.MetricsFunction('epm', snql_metric_layer=lambda args, alias: Function('rate', [Column(TransactionMRI.DURATION.value), args['interval'], 60], alias), optional_args=[fields.IntervalDefault('interval', 1, None)], default_result_type='rate'), fields.MetricsFunction('eps', snql_metric_layer=lambda args, alias: Function('rate', [Column(TransactionMRI.DURATION.value), args['interval'], 1], alias), optional_args=[fields.IntervalDefault('interval', 1, None)], default_result_type='rate'), fields.MetricsFunction('failure_count', snql_metric_layer=lambda args, alias: AliasedExpression(Column(TransactionMRI.FAILURE_COUNT.value), alias), default_result_type='integer'), fields.MetricsFunction('failure_rate', snql_metric_layer=lambda args, alias: AliasedExpression(Column(TransactionMRI.FAILURE_RATE.value), alias), default_result_type='percentage'), fields.MetricsFunction('http_error_count', snql_metric_layer=lambda args, alias: AliasedExpression(Column(TransactionMRI.HTTP_ERROR_COUNT.value), alias), default_result_type='integer'), fields.MetricsFunction('http_error_rate', snql_metric_layer=lambda args, alias: AliasedExpression(Column(TransactionMRI.HTTP_ERROR_RATE.value), alias), default_result_type='percentage'), fields.MetricsFunction('histogram', required_args=[fields.MetricArg('column')], snql_metric_layer=self._resolve_histogram_function, default_result_type='number', private=True)]}
        for (alias, name) in constants.FUNCTION_ALIASES.items():
            if name in function_converter:
                function_converter[alias] = function_converter[name].alias_as(alias)
        return function_converter

    def _resolve_title_alias(self, alias: str) -> SelectType:
        if False:
            return 10
        'title == transaction in discover'
        return AliasedExpression(self.builder.resolve_column('transaction'), alias)

    def _resolve_team_key_transaction_alias(self, _: str) -> SelectType:
        if False:
            for i in range(10):
                print('nop')
        team_key_transactions = field_aliases.get_team_transactions(self.builder)
        count = len(team_key_transactions)
        sentry_sdk.set_tag('team_key_txns.count', count)
        sentry_sdk.set_tag('team_key_txns.count.grouped', format_grouped_length(count, [10, 100, 250, 500]))
        if count == 0:
            team_key_transactions = [(-1, '')]
        return Function(function='team_key_transaction', parameters=[Column('e:transactions/team_key_transaction@none'), team_key_transactions], alias='team_key_transaction')

    def _event_type_converter(self, search_filter: SearchFilter) -> Optional[WhereType]:
        if False:
            return 10
        'Not really a converter, check its transaction, error otherwise'
        value = search_filter.value.value
        operator = search_filter.operator
        if value == 'transaction' and operator == '=':
            return None
        raise IncompatibleMetricsQuery('Can only filter event.type:transaction')

    def _project_slug_filter_converter(self, search_filter: SearchFilter) -> Optional[WhereType]:
        if False:
            print('Hello World!')
        return filter_aliases.project_slug_converter(self.builder, search_filter)

    def _release_filter_converter(self, search_filter: SearchFilter) -> Optional[WhereType]:
        if False:
            while True:
                i = 10
        return filter_aliases.release_filter_converter(self.builder, search_filter)

    def _transaction_filter_converter(self, search_filter: SearchFilter) -> Optional[WhereType]:
        if False:
            print('Hello World!')
        operator = search_filter.operator
        value = search_filter.value.value
        if operator in ('=', '!=') and value == '':
            if operator == '=':
                raise InvalidSearchQuery("All events have a transaction so this query wouldn't return anything")
            else:
                return None
        if isinstance(value, list):
            resolved_value = []
            for item in value:
                resolved_item = self.builder.resolve_tag_value(item)
                if resolved_item is None:
                    raise IncompatibleMetricsQuery(f'Transaction value {item} in filter not found')
                resolved_value.append(resolved_item)
        else:
            resolved_value = self.builder.resolve_tag_value(value)
            if resolved_value is None:
                raise IncompatibleMetricsQuery(f'Transaction value {value} in filter not found')
        value = resolved_value
        if search_filter.value.is_wildcard():
            return Condition(Function('match', [self.builder.resolve_column('transaction'), f'(?i){value}']), Op(search_filter.operator), 1)
        return Condition(self.builder.resolve_column('transaction'), Op(operator), value)

    def _resolve_apdex_function(self, args: Mapping[str, Union[str, Column, SelectType, int, float]], alias: Optional[str]=None) -> SelectType:
        if False:
            print('Hello World!')
        "Apdex is tag based in metrics, which means we can't base it on the satsifaction parameter"
        if args['satisfaction'] is not None:
            raise IncompatibleMetricsQuery('Cannot query apdex with a threshold parameter on the metrics dataset')
        return AliasedExpression(Column(TransactionMRI.APDEX.value), alias)

    def _resolve_user_misery_function(self, args: Mapping[str, Union[str, Column, SelectType, int, float]], alias: Optional[str]=None) -> SelectType:
        if False:
            for i in range(10):
                print('nop')
        if args['satisfaction'] is not None:
            raise IncompatibleMetricsQuery('Cannot query misery with a threshold parameter on the metrics dataset')
        return AliasedExpression(Column(TransactionMRI.USER_MISERY.value), alias)

    def _resolve_count_miserable_function(self, args: Mapping[str, Union[str, Column, SelectType, int, float]], alias: Optional[str]=None) -> SelectType:
        if False:
            while True:
                i = 10
        if args['satisfaction'] is not None:
            raise IncompatibleMetricsQuery('Cannot query misery with a threshold parameter on the metrics dataset')
        return AliasedExpression(Column(TransactionMRI.MISERABLE_USER.value), alias)

    def _key_transaction_filter_converter(self, search_filter: SearchFilter) -> Optional[WhereType]:
        if False:
            while True:
                i = 10
        return filter_aliases.team_key_transaction_filter(self.builder, search_filter)

    def _resolve_web_vital_function(self, args: Mapping[str, Union[str, Column, SelectType, int, float]], alias: str) -> SelectType:
        if False:
            i = 10
            return i + 15
        column = args['column']
        quality = args['quality'].lower()
        if column not in ['measurements.lcp', 'measurements.fcp', 'measurements.fp', 'measurements.fid', 'measurements.cls', 'measurements.ttfb']:
            raise InvalidSearchQuery('count_web_vitals only supports measurements')
        column = Column(constants.METRICS_MAP.get(column, column))
        if quality == 'any':
            return Function('count', [column], alias)
        return Function('count_web_vitals', [column, quality], alias)

    def _resolve_histogram_function(self, args: Mapping[str, Union[str, Column, SelectType, int, float]], alias: Optional[str]=None) -> SelectType:
        if False:
            i = 10
            return i + 15
        'zoom_params is based on running metrics zoom_histogram function that adds conditions based on min, max,\n        buckets'
        min_bin = getattr(self.builder, 'min_bin', None)
        max_bin = getattr(self.builder, 'max_bin', None)
        num_buckets = getattr(self.builder, 'num_buckets', 250)
        self.builder.histogram_aliases.append(alias)
        return Function('histogram', [Column(constants.METRICS_MAP.get(args['column'], args['column'])), min_bin, max_bin, num_buckets], alias)