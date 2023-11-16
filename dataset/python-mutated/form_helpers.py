from django import template
__all__ = ('getfield', 'render_custom_fields', 'render_errors', 'render_field', 'render_form', 'widget_type')
register = template.Library()

@register.filter()
def getfield(form, fieldname):
    if False:
        i = 10
        return i + 15
    '\n    Return the specified bound field of a Form.\n    '
    try:
        return form[fieldname]
    except KeyError:
        return None

@register.filter(name='widget_type')
def widget_type(field):
    if False:
        print('Hello World!')
    '\n    Return the widget type\n    '
    if hasattr(field, 'widget'):
        return field.widget.__class__.__name__.lower()
    elif hasattr(field, 'field'):
        return field.field.widget.__class__.__name__.lower()
    else:
        return None

@register.inclusion_tag('form_helpers/render_field.html')
def render_field(field, bulk_nullable=False, label=None):
    if False:
        return 10
    '\n    Render a single form field from template\n    '
    return {'field': field, 'label': label or field.label, 'bulk_nullable': bulk_nullable}

@register.inclusion_tag('form_helpers/render_custom_fields.html')
def render_custom_fields(form):
    if False:
        print('Hello World!')
    '\n    Render all custom fields in a form\n    '
    return {'form': form}

@register.inclusion_tag('form_helpers/render_form.html')
def render_form(form):
    if False:
        i = 10
        return i + 15
    '\n    Render an entire form from template\n    '
    return {'form': form}

@register.inclusion_tag('form_helpers/render_errors.html')
def render_errors(form):
    if False:
        for i in range(10):
            print('nop')
    '\n    Render form errors, if they exist.\n    '
    return {'form': form}