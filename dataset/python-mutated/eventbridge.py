from __future__ import annotations
import json
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook
from airflow.utils.helpers import prune_dict

def _validate_json(pattern: str) -> None:
    if False:
        print('Hello World!')
    try:
        json.loads(pattern)
    except ValueError:
        raise ValueError('`event_pattern` must be a valid JSON string.')

class EventBridgeHook(AwsBaseHook):
    """Amazon EventBridge Hook."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, client_type='events', **kwargs)

    def put_rule(self, name: str, description: str | None=None, event_bus_name: str | None=None, event_pattern: str | None=None, role_arn: str | None=None, schedule_expression: str | None=None, state: str | None=None, tags: list[dict] | None=None, **kwargs):
        if False:
            return 10
        '\n        Create or update an EventBridge rule.\n\n        :param name: name of the rule to create or update (required)\n        :param description: description of the rule\n        :param event_bus_name: name or ARN of the event bus to associate with this rule\n        :param event_pattern: pattern of events to be matched to this rule\n        :param role_arn: the Amazon Resource Name of the IAM role associated with the rule\n        :param schedule_expression: the scheduling expression (for example, a cron or rate expression)\n        :param state: indicates whether rule is set to be "ENABLED" or "DISABLED"\n        :param tags: list of key-value pairs to associate with the rule\n\n        '
        if not (event_pattern or schedule_expression):
            raise ValueError('One of `event_pattern` or `schedule_expression` are required in order to put or update your rule.')
        if state and state not in ['ENABLED', 'DISABLED']:
            raise ValueError('`state` must be specified as ENABLED or DISABLED.')
        if event_pattern:
            _validate_json(event_pattern)
        put_rule_kwargs: dict[str, str | list] = {**prune_dict({'Name': name, 'Description': description, 'EventBusName': event_bus_name, 'EventPattern': event_pattern, 'RoleArn': role_arn, 'ScheduleExpression': schedule_expression, 'State': state, 'Tags': tags})}
        return self.conn.put_rule(**put_rule_kwargs)