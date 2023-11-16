import copy
import typing
from typing import Any, Mapping
PARAMETERS_STR = '$parameters'
DEFAULT_MODEL_TYPES: Mapping[str, str] = {'CompositeErrorHandler.error_handlers': 'DefaultErrorHandler', 'CursorPagination.decoder': 'JsonDecoder', 'DatetimeBasedCursor.end_datetime': 'MinMaxDatetime', 'DatetimeBasedCursor.end_time_option': 'RequestOption', 'DatetimeBasedCursor.start_datetime': 'MinMaxDatetime', 'DatetimeBasedCursor.start_time_option': 'RequestOption', 'CustomIncrementalSync.end_datetime': 'MinMaxDatetime', 'CustomIncrementalSync.end_time_option': 'RequestOption', 'CustomIncrementalSync.start_datetime': 'MinMaxDatetime', 'CustomIncrementalSync.start_time_option': 'RequestOption', 'DeclarativeSource.check': 'CheckStream', 'DeclarativeSource.spec': 'Spec', 'DeclarativeSource.streams': 'DeclarativeStream', 'DeclarativeStream.retriever': 'SimpleRetriever', 'DeclarativeStream.schema_loader': 'JsonFileSchemaLoader', 'DefaultErrorHandler.response_filters': 'HttpResponseFilter', 'DefaultPaginator.decoder': 'JsonDecoder', 'DefaultPaginator.page_size_option': 'RequestOption', 'DpathExtractor.decoder': 'JsonDecoder', 'HttpRequester.error_handler': 'DefaultErrorHandler', 'ListPartitionRouter.request_option': 'RequestOption', 'ParentStreamConfig.request_option': 'RequestOption', 'ParentStreamConfig.stream': 'DeclarativeStream', 'RecordSelector.extractor': 'DpathExtractor', 'RecordSelector.record_filter': 'RecordFilter', 'SimpleRetriever.paginator': 'NoPagination', 'SimpleRetriever.record_selector': 'RecordSelector', 'SimpleRetriever.requester': 'HttpRequester', 'SubstreamPartitionRouter.parent_stream_configs': 'ParentStreamConfig', 'AddFields.fields': 'AddedFieldDefinition', 'CustomPartitionRouter.parent_stream_configs': 'ParentStreamConfig'}
CUSTOM_COMPONENTS_MAPPING: Mapping[str, str] = {'CompositeErrorHandler.backoff_strategies': 'CustomBackoffStrategy', 'DeclarativeStream.retriever': 'CustomRetriever', 'DeclarativeStream.transformations': 'CustomTransformation', 'DefaultErrorHandler.backoff_strategies': 'CustomBackoffStrategy', 'DefaultPaginator.pagination_strategy': 'CustomPaginationStrategy', 'HttpRequester.authenticator': 'CustomAuthenticator', 'HttpRequester.error_handler': 'CustomErrorHandler', 'RecordSelector.extractor': 'CustomRecordExtractor', 'SimpleRetriever.partition_router': 'CustomPartitionRouter'}

class ManifestComponentTransformer:

    def propagate_types_and_parameters(self, parent_field_identifier: str, declarative_component: Mapping[str, Any], parent_parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Recursively transforms the specified declarative component and subcomponents to propagate parameters and insert the\n        default component type if it was not already present. The resulting transformed components are a deep copy of the input\n        components, not an in-place transformation.\n\n        :param declarative_component: The current component that is having type and parameters added\n        :param parent_field_identifier: The name of the field of the current component coming from the parent component\n        :param parent_parameters: The parameters set on parent components defined before the current component\n        :return: A deep copy of the transformed component with types and parameters persisted to it\n        '
        propagated_component = dict(copy.deepcopy(declarative_component))
        if 'type' not in propagated_component:
            if 'class_name' in propagated_component:
                found_type = CUSTOM_COMPONENTS_MAPPING.get(parent_field_identifier)
            else:
                found_type = DEFAULT_MODEL_TYPES.get(parent_field_identifier)
            if found_type:
                propagated_component['type'] = found_type
        if 'type' not in propagated_component:
            return propagated_component
        current_parameters = dict(copy.deepcopy(parent_parameters))
        component_parameters = propagated_component.pop(PARAMETERS_STR, {})
        current_parameters = {**current_parameters, **component_parameters}
        for (parameter_key, parameter_value) in current_parameters.items():
            propagated_component[parameter_key] = propagated_component.get(parameter_key) or parameter_value
        for (field_name, field_value) in propagated_component.items():
            if isinstance(field_value, dict):
                excluded_parameter = current_parameters.pop(field_name, None)
                parent_type_field_identifier = f"{propagated_component.get('type')}.{field_name}"
                propagated_component[field_name] = self.propagate_types_and_parameters(parent_type_field_identifier, field_value, current_parameters)
                if excluded_parameter:
                    current_parameters[field_name] = excluded_parameter
            elif isinstance(field_value, typing.List):
                excluded_parameter = current_parameters.pop(field_name, None)
                for (i, element) in enumerate(field_value):
                    if isinstance(element, dict):
                        parent_type_field_identifier = f"{propagated_component.get('type')}.{field_name}"
                        field_value[i] = self.propagate_types_and_parameters(parent_type_field_identifier, element, current_parameters)
                if excluded_parameter:
                    current_parameters[field_name] = excluded_parameter
        if current_parameters:
            propagated_component[PARAMETERS_STR] = current_parameters
        return propagated_component