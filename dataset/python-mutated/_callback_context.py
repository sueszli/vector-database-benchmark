import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
context_value = contextvars.ContextVar('callback_context')
context_value.set({})

def has_context(func):
    if False:
        i = 10
        return i + 15

    @functools.wraps(func)
    def assert_context(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not context_value.get():
            raise exceptions.MissingCallbackContextException(f"dash.callback_context.{getattr(func, '__name__')} is only available from a callback!")
        return func(*args, **kwargs)
    return assert_context

def _get_context_value():
    if False:
        i = 10
        return i + 15
    return context_value.get()

class FalsyList(list):

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def __nonzero__(self):
        if False:
            for i in range(10):
                print('nop')
        return False
falsy_triggered = FalsyList([{'prop_id': '.', 'value': None}])

class CallbackContext:

    @property
    @has_context
    def inputs(self):
        if False:
            return 10
        return getattr(_get_context_value(), 'input_values', {})

    @property
    @has_context
    def states(self):
        if False:
            print('Hello World!')
        return getattr(_get_context_value(), 'state_values', {})

    @property
    @has_context
    def triggered(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns a list of all the Input props that changed and caused the callback to execute. It is empty when the\n        callback is called on initial load, unless an Input prop got its value from another initial callback.\n        Callbacks triggered by user actions typically have one item in triggered, unless the same action changes\n        two props at once or the callback has several Input props that are all modified by another callback based on\n        a single user action.\n\n        Example:  To get the id of the component that triggered the callback:\n        `component_id = ctx.triggered[0]['prop_id'].split('.')[0]`\n\n        Example:  To detect initial call, empty triggered is not really empty, it's falsy so that you can use:\n        `if ctx.triggered:`\n        "
        return getattr(_get_context_value(), 'triggered_inputs', []) or falsy_triggered

    @property
    @has_context
    def triggered_prop_ids(self):
        if False:
            while True:
                i = 10
        '\n        Returns a dictionary of all the Input props that changed and caused the callback to execute. It is empty when\n        the callback is called on initial load, unless an Input prop got its value from another initial callback.\n        Callbacks triggered by user actions typically have one item in triggered, unless the same action changes\n        two props at once or the callback has several Input props that are all modified by another callback based\n        on a single user action.\n\n        triggered_prop_ids (dict):\n        - keys (str) : the triggered "prop_id" composed of "component_id.component_property"\n        - values (str or dict): the id of the component that triggered the callback. Will be the dict id for pattern matching callbacks\n\n        Example - regular callback\n        {"btn-1.n_clicks": "btn-1"}\n\n        Example - pattern matching callbacks:\n        {\'{"index":0,"type":"filter-dropdown"}.value\': {"index":0,"type":"filter-dropdown"}}\n\n        Example usage:\n        `if "btn-1.n_clicks" in ctx.triggered_prop_ids:\n            do_something()`\n        '
        triggered = getattr(_get_context_value(), 'triggered_inputs', [])
        ids = AttributeDict({})
        for item in triggered:
            (component_id, _, _) = item['prop_id'].rpartition('.')
            ids[item['prop_id']] = component_id
            if component_id.startswith('{'):
                ids[item['prop_id']] = AttributeDict(json.loads(component_id))
        return ids

    @property
    @has_context
    def triggered_id(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the component id (str or dict) of the Input component that triggered the callback.\n\n        Note - use `triggered_prop_ids` if you need both the component id and the prop that triggered the callback or if\n        multiple Inputs triggered the callback.\n\n        Example usage:\n        `if "btn-1" == ctx.triggered_id:\n            do_something()`\n\n        '
        component_id = None
        if self.triggered:
            prop_id = self.triggered_prop_ids.first()
            component_id = self.triggered_prop_ids[prop_id]
        return component_id

    @property
    @has_context
    def args_grouping(self):
        if False:
            print('Hello World!')
        '\n        args_grouping is a dict of the inputs used with flexible callback signatures. The keys are the variable names\n        and the values are dictionaries containing:\n        - “id”: (string or dict) the component id. If it’s a pattern matching id, it will be a dict.\n        - “id_str”: (str) for pattern matching ids, it’s the stringified dict id with no white spaces.\n        - “property”: (str) The component property used in the callback.\n        - “value”: the value of the component property at the time the callback was fired.\n        - “triggered”: (bool)Whether this input triggered the callback.\n\n        Example usage:\n        @app.callback(\n            Output("container", "children"),\n            inputs=dict(btn1=Input("btn-1", "n_clicks"), btn2=Input("btn-2", "n_clicks")),\n        )\n        def display(btn1, btn2):\n            c = ctx.args_grouping\n            if c.btn1.triggered:\n                return f"Button 1 clicked {btn1} times"\n            elif c.btn2.triggered:\n                return f"Button 2 clicked {btn2} times"\n            else:\n               return "No clicks yet"\n\n        '
        return getattr(_get_context_value(), 'args_grouping', [])

    @property
    @has_context
    def outputs_grouping(self):
        if False:
            return 10
        return getattr(_get_context_value(), 'outputs_grouping', [])

    @property
    @has_context
    def outputs_list(self):
        if False:
            while True:
                i = 10
        if self.using_outputs_grouping:
            warnings.warn('outputs_list is deprecated, use outputs_grouping instead', DeprecationWarning)
        return getattr(_get_context_value(), 'outputs_list', [])

    @property
    @has_context
    def inputs_list(self):
        if False:
            for i in range(10):
                print('nop')
        if self.using_args_grouping:
            warnings.warn('inputs_list is deprecated, use args_grouping instead', DeprecationWarning)
        return getattr(_get_context_value(), 'inputs_list', [])

    @property
    @has_context
    def states_list(self):
        if False:
            print('Hello World!')
        if self.using_args_grouping:
            warnings.warn('states_list is deprecated, use args_grouping instead', DeprecationWarning)
        return getattr(_get_context_value(), 'states_list', [])

    @property
    @has_context
    def response(self):
        if False:
            print('Hello World!')
        return getattr(_get_context_value(), 'dash_response')

    @staticmethod
    @has_context
    def record_timing(name, duration=None, description=None):
        if False:
            print('Hello World!')
        'Records timing information for a server resource.\n\n        :param name: The name of the resource.\n        :type name: string\n\n        :param duration: The time in seconds to report. Internally, this\n            is rounded to the nearest millisecond.\n        :type duration: float or None\n\n        :param description: A description of the resource.\n        :type description: string or None\n        '
        timing_information = getattr(flask.g, 'timing_information', {})
        if name in timing_information:
            raise KeyError(f'Duplicate resource name "{name}" found.')
        timing_information[name] = {'dur': round(duration * 1000), 'desc': description}
        setattr(flask.g, 'timing_information', timing_information)

    @property
    @has_context
    def using_args_grouping(self):
        if False:
            while True:
                i = 10
        '\n        Return True if this callback is using dictionary or nested groupings for\n        Input/State dependencies, or if Input and State dependencies are interleaved\n        '
        return getattr(_get_context_value(), 'using_args_grouping', [])

    @property
    @has_context
    def using_outputs_grouping(self):
        if False:
            i = 10
            return i + 15
        '\n        Return True if this callback is using dictionary or nested groupings for\n        Output dependencies.\n        '
        return getattr(_get_context_value(), 'using_outputs_grouping', [])

    @property
    @has_context
    def timing_information(self):
        if False:
            for i in range(10):
                print('nop')
        return getattr(flask.g, 'timing_information', {})
callback_context = CallbackContext()