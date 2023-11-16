"""Utility to re-use variables created on first device on subsequent devices."""
import re
_VARIABLE_UNIQUIFYING_REGEX = re.compile('_\\d/')
_VARIABLE_UNIQUIFYING_REGEX_AT_END = re.compile('_\\d$')

def _canonicalize_variable_name(name):
    if False:
        while True:
            i = 10
    if name is None:
        return 'Variable'
    name = _VARIABLE_UNIQUIFYING_REGEX.sub('/', name)
    name = _VARIABLE_UNIQUIFYING_REGEX_AT_END.sub('', name)
    return name

def make_fn(shared_variable_store, device_id):
    if False:
        for i in range(10):
            print('nop')
    'Construct the variable creator function for device `device_id`.\n\n  Constructs custom variable creator functions for the given device.\n  On first device (device_id == 0), it creates the variable using the\n  `next_creator`, and stores it in the provided `shared_variable_store`.\n  On all other devices (device_id > 0), it tries to re-use the variable\n  already created with the same name. If no such variable exists, it throws an\n  error.\n  Additionally, we de-uniquify variable names before checking for matches. This\n  helps re-use variables which are intended to be the same but have different\n  names due to variable uniquification happening upstream. Since this might\n  mean we may have multiple variables with the same canonical name, we store\n  them in a list per canonical name and return them in the same order as well.\n\n  Args:\n    shared_variable_store: A dictionary that we will use to store variables\n      created on the first device, and re-used by creators for other devices.\n    device_id: Integer index of the device whose creator should be\n      constructed.\n\n  Returns:\n    An appropriate creator function based on device_id.\n\n  '
    variable_scope_access_index = {}
    assert isinstance(device_id, int)

    def create_new_variable(next_creator, **kwargs):
        if False:
            while True:
                i = 10
        'Create the variable using `next_creator` and store it.'
        canonical_name = _canonicalize_variable_name(kwargs.get('name'))
        v = next_creator(**kwargs)
        if canonical_name not in shared_variable_store:
            shared_variable_store[canonical_name] = []
        shared_variable_store[canonical_name].append(v)
        return v

    def reuse_variable(next_creator, **kwargs):
        if False:
            return 10
        'Re-use existing variable from store with same name (in order).'
        del next_creator
        name = kwargs.get('name')
        canonical_name = _canonicalize_variable_name(name)
        try:
            variable_index = variable_scope_access_index.get(canonical_name, 0)
            v = shared_variable_store[canonical_name][variable_index]
            variable_scope_access_index[canonical_name] = variable_index + 1
            return v
        except (KeyError, IndexError):
            raise RuntimeError('Tried to create variable {} with mismatching name on device {}'.format(name, device_id))
    if device_id == 0:
        return create_new_variable
    else:
        return reuse_variable