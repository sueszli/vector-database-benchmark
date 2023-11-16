from itertools import chain

def get_all_field_names(model):
    if False:
        print('Hello World!')
    return list(set(chain.from_iterable(((field.name, field.attname) if hasattr(field, 'attname') else (field.name,) for field in model._meta.get_fields() if not (field.many_to_one and field.related_model is None)))))