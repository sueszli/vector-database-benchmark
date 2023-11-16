from __future__ import annotations
from ansible.errors import AnsibleError
from ansible.plugins.action import ActionBase
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.utils.vars import combine_vars

class ActionModule(ActionBase):
    """ Validate an arg spec"""
    TRANSFERS_FILES = False
    _requires_connection = False

    def get_args_from_task_vars(self, argument_spec, task_vars):
        if False:
            while True:
                i = 10
        '\n        Get any arguments that may come from `task_vars`.\n\n        Expand templated variables so we can validate the actual values.\n\n        :param argument_spec: A dict of the argument spec.\n        :param task_vars: A dict of task variables.\n\n        :returns: A dict of values that can be validated against the arg spec.\n        '
        args = {}
        for (argument_name, argument_attrs) in argument_spec.items():
            if argument_name in task_vars:
                args[argument_name] = task_vars[argument_name]
        args = self._templar.template(args)
        return args

    def run(self, tmp=None, task_vars=None):
        if False:
            while True:
                i = 10
        "\n        Validate an argument specification against a provided set of data.\n\n        The `validate_argument_spec` module expects to receive the arguments:\n            - argument_spec: A dict whose keys are the valid argument names, and\n                  whose values are dicts of the argument attributes (type, etc).\n            - provided_arguments: A dict whose keys are the argument names, and\n                  whose values are the argument value.\n\n        :param tmp: Deprecated. Do not use.\n        :param task_vars: A dict of task variables.\n        :return: An action result dict, including a 'argument_errors' key with a\n            list of validation errors found.\n        "
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        result['validate_args_context'] = self._task.args.get('validate_args_context', {})
        if 'argument_spec' not in self._task.args:
            raise AnsibleError('"argument_spec" arg is required in args: %s' % self._task.args)
        argument_spec_data = self._task.args.get('argument_spec')
        provided_arguments = self._task.args.get('provided_arguments', {})
        if not isinstance(argument_spec_data, dict):
            raise AnsibleError('Incorrect type for argument_spec, expected dict and got %s' % type(argument_spec_data))
        if not isinstance(provided_arguments, dict):
            raise AnsibleError('Incorrect type for provided_arguments, expected dict and got %s' % type(provided_arguments))
        args_from_vars = self.get_args_from_task_vars(argument_spec_data, task_vars)
        validator = ArgumentSpecValidator(argument_spec_data)
        validation_result = validator.validate(combine_vars(args_from_vars, provided_arguments), validate_role_argument_spec=True)
        if validation_result.error_messages:
            result['failed'] = True
            result['msg'] = 'Validation of arguments failed:\n%s' % '\n'.join(validation_result.error_messages)
            result['argument_spec_data'] = argument_spec_data
            result['argument_errors'] = validation_result.error_messages
            return result
        result['changed'] = False
        result['msg'] = 'The arg spec validation passed'
        return result