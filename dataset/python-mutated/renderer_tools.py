import traceback
import json
from .basic import SerializationErrorComponent

def _render_component_safely(component, render_func, return_error_component, *args, **kwargs):
    if False:
        print('Hello World!')
    rendered_obj = None
    try:
        rendered_obj = render_func(component, *args, **kwargs)
    except:
        if not return_error_component:
            return None
        error_str = traceback.format_exc()
        rendered_obj = SerializationErrorComponent(component.__class__.__name__, error_str).render()
    else:
        if not (type(rendered_obj) == str or type(rendered_obj) == dict):
            rendered_obj = SerializationErrorComponent(component.__class__.__name__, "Component render didn't return a `string` or `dict`").render()
        else:
            try:
                json.dumps(rendered_obj)
            except (TypeError, OverflowError) as e:
                rendered_obj = SerializationErrorComponent(component.__class__.__name__, 'Rendered Component cannot be JSON serialized. \n\n %s' % str(e)).render()
    return rendered_obj

def render_safely(func):
    if False:
        while True:
            i = 10
    '\n    This is a decorator that can be added to any `MetaflowCardComponent.render`\n    The goal is to render subcomponents safely and ensure that they are JSON serializable.\n    '

    def ret_func(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return _render_component_safely(self, func, True, *args, **kwargs)
    return ret_func