from django.template import Library
register = Library()

@register.filter(name='bootstrap_field')
def bootstrap_field(field, class_attr):
    if False:
        for i in range(10):
            print('nop')
    return field.as_widget(attrs={'placeholder': field.label, 'class': class_attr})