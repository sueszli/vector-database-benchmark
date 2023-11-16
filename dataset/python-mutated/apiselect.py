import json
from typing import Dict, List, Tuple
from django import forms
from django.conf import settings
__all__ = ('APISelect', 'APISelectMultiple')

class APISelect(forms.Select):
    """
    A select widget populated via an API call

    :param api_url: API endpoint URL. Required if not set automatically by the parent field.
    """
    template_name = 'widgets/apiselect.html'
    option_template_name = 'widgets/select_option.html'
    dynamic_params: Dict[str, str]
    static_params: Dict[str, List[str]]

    def __init__(self, api_url=None, full=False, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.attrs['class'] = 'netbox-api-select'
        self.dynamic_params: Dict[str, List[str]] = {}
        self.static_params: Dict[str, List[str]] = {}
        if api_url:
            self.attrs['data-url'] = '/{}{}'.format(settings.BASE_PATH, api_url.lstrip('/'))

    def __deepcopy__(self, memo):
        if False:
            return 10
        'Reset `static_params` and `dynamic_params` when APISelect is deepcopied.'
        result = super().__deepcopy__(memo)
        result.dynamic_params = {}
        result.static_params = {}
        return result

    def _process_query_param(self, key, value) -> None:
        if False:
            return 10
        "\n        Based on query param value's type and value, update instance's dynamic/static params.\n        "
        if isinstance(value, str):
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value is None:
                value = 'null'
        if isinstance(value, str):
            if value.startswith('$'):
                field_name = value.strip('$')
                self.dynamic_params[field_name] = key
            elif key in self.static_params:
                current = self.static_params[key]
                self.static_params[key] = [v for v in set([*current, value])]
            else:
                self.static_params[key] = [value]
        elif key in self.static_params:
            current = self.static_params[key]
            self.static_params[key] = [v for v in set([*current, value])]
        else:
            self.static_params[key] = [value]

    def _process_query_params(self, query_params):
        if False:
            print('Hello World!')
        '\n        Process an entire query_params dictionary, and handle primitive or list values.\n        '
        for (key, value) in query_params.items():
            if isinstance(value, (List, Tuple)):
                for item in value:
                    self._process_query_param(key, item)
            else:
                self._process_query_param(key, value)

    def _serialize_params(self, key, params):
        if False:
            return 10
        '\n        Serialize dynamic or static query params to JSON and add the serialized value to\n        the widget attributes by `key`.\n        '
        current = json.loads(self.attrs.get(key, '[]'))
        self.attrs[key] = json.dumps([*current, *params], separators=(',', ':'))

    def _add_dynamic_params(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert post-processed dynamic query params to data structure expected by front-\n        end, serialize the value to JSON, and add it to the widget attributes.\n        '
        key = 'data-dynamic-params'
        if len(self.dynamic_params) > 0:
            try:
                update = [{'fieldName': f, 'queryParam': q} for (f, q) in self.dynamic_params.items()]
                self._serialize_params(key, update)
            except IndexError as error:
                raise RuntimeError(f"Missing required value for dynamic query param: '{self.dynamic_params}'") from error

    def _add_static_params(self):
        if False:
            print('Hello World!')
        '\n        Convert post-processed static query params to data structure expected by front-\n        end, serialize the value to JSON, and add it to the widget attributes.\n        '
        key = 'data-static-params'
        if len(self.static_params) > 0:
            try:
                update = [{'queryParam': k, 'queryValue': v} for (k, v) in self.static_params.items()]
                self._serialize_params(key, update)
            except IndexError as error:
                raise RuntimeError(f"Missing required value for static query param: '{self.static_params}'") from error

    def add_query_params(self, query_params):
        if False:
            while True:
                i = 10
        '\n        Proccess & add a dictionary of URL query parameters to the widget attributes.\n        '
        self._process_query_params(query_params)
        self._add_dynamic_params()
        self._add_static_params()

    def add_query_param(self, key, value) -> None:
        if False:
            return 10
        '\n        Process & add a key/value pair of URL query parameters to the widget attributes.\n        '
        self.add_query_params({key: value})

class APISelectMultiple(APISelect, forms.SelectMultiple):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.attrs['data-multiple'] = 1