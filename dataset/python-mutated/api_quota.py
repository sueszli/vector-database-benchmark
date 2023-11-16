import logging
from functools import wraps
from typing import Any, Iterable, Mapping, Optional
import requests
from requests.exceptions import JSONDecodeError
from .utils import API_LIMIT_PER_HOUR

class GoogleAnalyticsApiQuotaBase:
    logger = logging.getLogger('airbyte')
    initial_quota: Optional[Mapping[str, Any]] = None
    treshold: float = 0.1
    should_retry: Optional[bool] = True
    backoff_time: Optional[int] = None
    raise_on_http_errors: bool = True
    stop_iter: bool = False
    error_message = None
    quota_mapping: Mapping[str, Any] = {'concurrentRequests': {'error_pattern': 'Exhausted concurrent requests quota.', 'backoff': 30, 'should_retry': True, 'raise_on_http_errors': False, 'stop_iter': False}, 'tokensPerProjectPerHour': {'error_pattern': 'Exhausted property tokens for a project per hour.', 'backoff': 1800, 'should_retry': True, 'raise_on_http_errors': False, 'stop_iter': False, 'error_message': API_LIMIT_PER_HOUR}, 'potentiallyThresholdedRequestsPerHour': {'error_pattern': 'Exhausted potentially thresholded requests quota.', 'backoff': 1800, 'should_retry': True, 'raise_on_http_errors': False, 'stop_iter': False, 'error_message': API_LIMIT_PER_HOUR}}

    def _get_known_quota_list(self) -> Iterable[str]:
        if False:
            print('Hello World!')
        return self.quota_mapping.keys()

    def _get_initial_quota_value(self, quota_name: str) -> int:
        if False:
            i = 10
            return i + 15
        init_remaining = self.initial_quota.get(quota_name).get('remaining')
        return 1 if init_remaining <= 0 else init_remaining

    def _get_quota_name_from_error_message(self, error_msg: str) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        for (quota, value) in self.quota_mapping.items():
            if value.get('error_pattern') in error_msg:
                return quota
        return None

    def _get_known_quota_from_response(self, property_quota: Mapping[str, Any]) -> Mapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        current_quota = {}
        for quota in property_quota.keys():
            if quota in self._get_known_quota_list():
                current_quota.update(**{quota: property_quota.get(quota)})
        return current_quota

    def _set_retry_attrs_for_quota(self, quota_name: str) -> None:
        if False:
            return 10
        quota = self.quota_mapping.get(quota_name, {})
        if quota:
            self.should_retry = quota.get('should_retry')
            self.raise_on_http_errors = quota.get('raise_on_http_errors')
            self.stop_iter = quota.get('stop_iter')
            self.backoff_time = quota.get('backoff')
            self.error_message = quota.get('error_message')

    def _set_default_retry_attrs(self) -> None:
        if False:
            i = 10
            return i + 15
        self.should_retry = True
        self.backoff_time = None
        self.raise_on_http_errors = True
        self.stop_iter = False

    def _set_initial_quota(self, current_quota: Optional[Mapping[str, Any]]=None) -> None:
        if False:
            print('Hello World!')
        if not self.initial_quota:
            self.initial_quota = current_quota

    def _check_remaining_quota(self, current_quota: Mapping[str, Any]) -> None:
        if False:
            print('Hello World!')
        for (quota_name, quota_value) in current_quota.items():
            total_available = self._get_initial_quota_value(quota_name)
            remaining: int = quota_value.get('remaining')
            remaining_percent: float = remaining / total_available
            if remaining_percent <= self.treshold:
                self.logger.warning(f'The `{quota_name}` quota is running out of tokens. Available {remaining} out of {total_available}.')
                self._set_retry_attrs_for_quota(quota_name)
                return None
            elif self.error_message:
                self.logger.warning(self.error_message)

    def _check_for_errors(self, response: requests.Response) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            self._set_default_retry_attrs()
            error = response.json().get('error')
            if error:
                quota_name = self._get_quota_name_from_error_message(error.get('message'))
                if quota_name:
                    self._set_retry_attrs_for_quota(quota_name)
                    self.logger.warn(f'The `{quota_name}` quota is exceeded!')
                    return None
        except (AttributeError, JSONDecodeError) as attr_e:
            self.logger.warning(f'`GoogleAnalyticsApiQuota._check_for_errors`: Received non JSON response from the API. Full error: {attr_e}. Bypassing.')
            pass
        except Exception as e:
            self.logger.fatal(f'Other `GoogleAnalyticsApiQuota` error: {e}')
            raise

class GoogleAnalyticsApiQuota(GoogleAnalyticsApiQuotaBase):

    def _check_quota(self, response: requests.Response):
        if False:
            while True:
                i = 10
        try:
            parsed_response = response.json()
        except (AttributeError, JSONDecodeError) as e:
            self.logger.warn(f'`GoogleAnalyticsApiQuota._check_quota`: Received non JSON response from the API. Full error: {e}. Bypassing.')
            parsed_response = {}
        property_quota: dict = parsed_response.get('propertyQuota')
        if property_quota:
            self._set_default_retry_attrs()
            current_quota = self._get_known_quota_from_response(property_quota)
            if current_quota:
                self._set_initial_quota(current_quota)
                self._check_remaining_quota(current_quota)
        else:
            self._check_for_errors(response)

    def handle_quota(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The function decorator is used to integrate with the `should_retry` method,\n        or any other method that provides early access to the `response` object.\n        '

        def decorator(func):
            if False:
                i = 10
                return i + 15

            @wraps(func)
            def wrapper_handle_quota(*args, **kwargs):
                if False:
                    print('Hello World!')
                for arg in args:
                    response = arg if isinstance(arg, requests.models.Response) else None
                self._check_quota(response)
                return func(*args, **kwargs)
            return wrapper_handle_quota
        return decorator