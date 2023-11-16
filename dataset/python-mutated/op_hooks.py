from unittest import mock
import yaml
from dagster_slack import slack_resource
from dagster import HookContext, ResourceDefinition, failure_hook, file_relative_path, graph, job, op, repository, success_hook

@success_hook(required_resource_keys={'slack'})
def slack_message_on_success(context: HookContext):
    if False:
        return 10
    message = f'Op {context.op.name} finished successfully'
    context.resources.slack.chat_postMessage(channel='#foo', text=message)

@failure_hook(required_resource_keys={'slack'})
def slack_message_on_failure(context: HookContext):
    if False:
        print('Hello World!')
    message = f'Op {context.op.name} failed'
    context.resources.slack.chat_postMessage(channel='#foo', text=message)
slack_resource_mock = mock.MagicMock()

@op
def a():
    if False:
        while True:
            i = 10
    pass

@op
def b():
    if False:
        i = 10
        return i + 15
    raise Exception()

@job(resource_defs={'slack': slack_resource}, hooks={slack_message_on_failure})
def notif_all():
    if False:
        return 10
    a()
    b()

@graph
def slack_notif_all():
    if False:
        return 10
    a()
    b()
notif_all_dev = slack_notif_all.to_job(name='notif_all_dev', resource_defs={'slack': ResourceDefinition.hardcoded_resource(slack_resource_mock, 'do not send messages in dev')}, hooks={slack_message_on_failure})
notif_all_prod = slack_notif_all.to_job(name='notif_all_prod', resource_defs={'slack': slack_resource}, hooks={slack_message_on_failure})

@job(resource_defs={'slack': slack_resource})
def selective_notif():
    if False:
        while True:
            i = 10
    a.with_hooks({slack_message_on_failure, slack_message_on_success})()
    b()

@repository
def repo():
    if False:
        for i in range(10):
            print('nop')
    return [notif_all, selective_notif]
if __name__ == '__main__':
    prod_op_hooks_run_config_yaml = file_relative_path(__file__, 'prod_op_hooks.yaml')
    with open(prod_op_hooks_run_config_yaml, 'r', encoding='utf8') as fd:
        run_config = yaml.safe_load(fd.read())
    notif_all_prod.execute_in_process(run_config=run_config, raise_on_error=False)
from dagster import build_hook_context

@success_hook(required_resource_keys={'my_conn'})
def my_success_hook(context):
    if False:
        print('Hello World!')
    context.resources.my_conn.send('foo')

def test_my_success_hook():
    if False:
        i = 10
        return i + 15
    my_conn = mock.MagicMock()
    context = build_hook_context(resources={'my_conn': my_conn})
    my_success_hook(context)
    assert my_conn.send.call_count == 1

@job(resource_defs={'slack': slack_resource.configured({'token': 'xoxp-1234123412341234-12341234-1234'})}, hooks={slack_message_on_failure})
def notif_all_configured():
    if False:
        while True:
            i = 10
    a()
    b()