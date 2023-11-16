from __future__ import absolute_import
import argparse
__all__ = ['generate_argument_parser_for_metadata']

def generate_argument_parser_for_metadata(metadata):
    if False:
        i = 10
        return i + 15
    '\n    Generate ArgumentParser instance for the action with the provided metadata\n    object.\n\n    :param metadata: Action metadata\n    :type metadata: ``dict``\n\n    :return: Generated argument parser instance.\n    :rtype: :class:`argparse.ArgumentParser`\n    '
    parameters = metadata['parameters']
    parser = argparse.ArgumentParser(description=metadata['description'])
    for (parameter_name, parameter_options) in parameters.items():
        name = parameter_name.replace('_', '-')
        description = parameter_options['description']
        _type = parameter_options['type']
        required = parameter_options.get('required', False)
        default_value = parameter_options.get('default', None)
        immutable = parameter_options.get('immutable', False)
        if immutable:
            continue
        args = ['--%s' % name]
        kwargs = {'help': description, 'required': required}
        if default_value is not None:
            kwargs['default'] = default_value
        if _type == 'string':
            kwargs['type'] = str
        elif _type == 'integer':
            kwargs['type'] = int
        elif _type == 'boolean':
            if default_value is False:
                kwargs['action'] = 'store_false'
            else:
                kwargs['action'] = 'store_true'
        parser.add_argument(*args, **kwargs)
    return parser