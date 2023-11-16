from __future__ import absolute_import, division, print_function
__metaclass__ = type
from ansible.module_utils.basic import AnsibleModule
from lwe.core.config import Config
from lwe import ApiBackend
import lwe.core.util as util
DOCUMENTATION = '\n---\nmodule: lwe_llm\n\nshort_description: Make LLM requests via LWE.\n\nversion_added: "1.0.0"\n\ndescription: Make LLM requests via LWE.\n\noptions:\n    message:\n        description: The message to send to the model.\n        required: true if template not provided\n        type: str\n    profile:\n        description: The LWE profile to use.\n        required: false\n        default: \'default\'\n        type: str\n    preset:\n        description: The LWE preset to use.\n        required: false\n        default: None\n        type: str\n    preset_overrides:\n        description: A dictionary of metadata and model customization overrides to apply to the preset when running the template.\n        required: false\n        default: None\n        type: dict\n    system_message:\n        description: The LWE system message to use, either an alias or custom message.\n        required: false\n        default: None\n        type: str\n    max_submission_tokens:\n        description: The maximum number of tokens that can be submitted. Default is max for the model.\n        required: false\n        default: None\n        type: int\n    template:\n        description: An LWE template to use for constructing the prompt.\n        required: true if message not provided\n        default: None\n        type: str\n    template_vars:\n        description: A dictionary of variables to substitute into the template.\n        required: false\n        default: None\n        type: dict\n    user:\n        description: The LWE user to load for the execution, a user ID or username.\n                     NOTE: A user must be provided to start or continue a conversation.\n        required: false\n        default: None (anonymous)\n        type: str\n    conversation_id:\n        description: An existing LWE conversation to use.\n                     NOTE: A conversation_id must be provided to continue a conversation.\n        required: false\n        default: None (anonymous, or new conversation if user is provided)\n        type: int\n\nauthor:\n    - Chad Phillips (@thehunmonkgroup)\n'
EXAMPLES = '\n# Simple message with default values\n- name: Say hello\n  lwe_llm:\n    message: "Say Hello!"\n\n# Start a new conversation with this response\n- name: Start conversation\n  lwe_llm:\n    message: "What are the three primary colors?"\n    max_submission_tokens: 512\n    # User ID or username\n    user: 1\n    register: result\n\n# Continue a conversation with this response\n- name: Continue conversation\n  lwe_llm:\n    message: "Provide more detail about your previous response"\n    user: 1\n    conversation_id: result.conversation_id\n\n# Use the \'mytemplate.md\' template, passing in a few template variables\n- name: Templated prompt\n  lwe_llm:\n    template: mytemplate.md\n    template_vars:\n        foo: bar\n        baz: bang\n\n# Use the \'test\' profile, a pre-configured provider/model preset \'mypreset\',\n# and override some of the preset configuration.\n- name: Continue conversation\n  lwe_llm:\n    message: "Say three things about bacon"\n    system_message: "You are a bacon connoisseur"\n    profile: test\n    preset: mypreset\n    preset_overrides:\n        metadata:\n            return_on_function_call: true\n        model_customizations:\n            temperature: 1\n\n'
RETURN = '\nresponse:\n    description: The response from the model.\n    type: str\n    returned: always\nconversation_id:\n    description: The conversation ID if the task run is associated with a conversation, or None otherwise.\n    type: int\n    returned: always\nuser_message:\n    description: Human-readable user status message for the response.\n    type: str\n    returned: always\n'

def run_module():
    if False:
        return 10
    module_args = dict(message=dict(type='str', required=False), profile=dict(type='str', required=False, default='default'), preset=dict(type='str', required=False), preset_overrides=dict(type='dict', required=False), system_message=dict(type='str', required=False), max_submission_tokens=dict(type='int', required=False), template=dict(type='str', required=False), template_vars=dict(type='dict', required=False), user=dict(type='raw', required=False), conversation_id=dict(type='int', required=False))
    result = dict(changed=False, response=dict())
    module = AnsibleModule(argument_spec=module_args, supports_check_mode=True)
    message = module.params['message']
    profile = module.params['profile']
    preset = module.params['preset']
    preset_overrides = module.params['preset_overrides']
    system_message = module.params['system_message']
    max_submission_tokens = module.params['max_submission_tokens']
    template_name = module.params['template']
    template_vars = module.params['template_vars'] or {}
    user = module.params['user']
    try:
        user = int(user)
    except Exception:
        pass
    conversation_id = module.params['conversation_id']
    if message is None and template_name is None or (message is not None and template_name is not None):
        module.fail_json(msg="One and only one of 'message' or 'template' arguments must be set.")
    if module.check_mode:
        module.exit_json(**result)
    config = Config(profile=profile)
    config.load_from_file()
    config.set('debug.log.enabled', True)
    config.set('model.default_preset', preset)
    config.set('backend_options.default_user', user)
    config.set('backend_options.default_conversation_id', conversation_id)
    gpt = ApiBackend(config)
    if max_submission_tokens:
        gpt.set_max_submission_tokens(max_submission_tokens)
    gpt.set_return_only(True)
    gpt.log.info('[lwe_llm module]: Starting execution')
    overrides = {'request_overrides': {}}
    if preset_overrides:
        overrides['request_overrides']['preset_overrides'] = preset_overrides
    if system_message:
        overrides['request_overrides']['system_message'] = system_message
    if template_name is not None:
        gpt.log.debug(f'[lwe_llm module]: Using template: {template_name}')
        (success, response, user_message) = gpt.template_manager.get_template_variables_substitutions(template_name)
        if not success:
            gpt.log.error(f'[lwe_llm module]: {user_message}')
            module.fail_json(msg=user_message, **result)
        (_template, _variables, substitutions) = response
        util.merge_dicts(substitutions, template_vars)
        (success, response, user_message) = gpt.run_template_setup(template_name, substitutions)
        if not success:
            gpt.log.error(f'[lwe_llm module]: {user_message}')
            module.fail_json(msg=user_message, **result)
        (message, template_overrides) = response
        util.merge_dicts(template_overrides, overrides)
        gpt.log.info(f'[lwe_llm module]: Running template: {template_name}')
        (success, response, user_message) = gpt.run_template_compiled(message, template_overrides)
        if not success:
            gpt.log.error(f'[lwe_llm module]: {user_message}')
            module.fail_json(msg=user_message, **result)
    else:
        (success, response, user_message) = gpt.ask(message, **overrides)
    if not success or not response:
        result['failed'] = True
        message = user_message
        if not success:
            message = f'Error fetching LLM response: {user_message}'
        elif not response:
            message = f'Empty LLM response: {user_message}'
        gpt.log.error(f'[lwe_llm module]: {message}')
        module.fail_json(msg=message, **result)
    result['changed'] = True
    result['response'] = response
    result['conversation_id'] = gpt.conversation_id
    result['user_message'] = user_message
    gpt.log.info('[lwe_llm module]: execution completed successfully')
    module.exit_json(**result)

def main():
    if False:
        for i in range(10):
            print('nop')
    run_module()
if __name__ == '__main__':
    main()