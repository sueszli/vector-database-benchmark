import six
from mongoengine.queryset import Q
from st2common import log as logging
from st2api.controllers.resource import BaseResourceIsolationControllerMixin
from st2api.controllers.resource import ContentPackResourceController
from st2common.models.api.rule import RuleViewAPI
from st2common.models.system.common import ResourceReference
from st2common.persistence.action import Action
from st2common.persistence.rule import Rule
from st2common.persistence.trigger import TriggerType, Trigger
from st2common.rbac.types import PermissionType
http_client = six.moves.http_client
LOG = logging.getLogger(__name__)
__all__ = ['RuleViewController']

class RuleViewController(BaseResourceIsolationControllerMixin, ContentPackResourceController):
    """
    Add some extras to a Rule object to make it easier for UI to render a rule. The additions
    do not necessarily belong in the Rule itself but are still valuable augmentations.

    :Example:
        {
            "action": {
                "description": "Action that executes an arbitrary Linux command on the localhost.",
                "parameters": {
                    "cmd": "echo "{{trigger.executed_at}}""
                },
                "ref": "core.local"
            },
            "criteria": {},
            "description": "Sample rule using an Interval Timer.",
            "enabled": false,
            "id": "55ea221832ed35759cf3b312",
            "name": "sample.with_timer",
            "pack": "examples",
            "ref": "examples.sample.with_timer",
            "tags": [],
            "trigger": {
                "description": "Triggers on specified intervals. e.g. every 30s, 1week etc.",
                "parameters": {
                    "delta": 5,
                    "unit": "seconds"
                },
                "ref": "core.4ad65602-6fb4-4c89-b0f2-b990d7b68bad",
                "type": "core.st2.IntervalTimer"
            },
            "uid": "rule:examples:sample.with_timer"
        }

    The `description` fields in action and trigger are augmented properties.
    """
    model = RuleViewAPI
    access = Rule
    supported_filters = {'name': 'name', 'pack': 'pack', 'user': 'context.user'}
    query_options = {'sort': ['pack', 'name']}
    mandatory_include_fields_retrieve = ['pack', 'name', 'trigger']

    def get_all(self, exclude_attributes=None, include_attributes=None, sort=None, offset=0, limit=None, requester_user=None, **raw_filters):
        if False:
            for i in range(10):
                print('nop')
        rules = super(RuleViewController, self)._get_all(exclude_fields=exclude_attributes, include_fields=include_attributes, sort=sort, offset=offset, limit=limit, raw_filters=raw_filters, requester_user=requester_user)
        result = self._append_view_properties(rules.json)
        rules.json = result
        return rules

    def get_one(self, ref_or_id, requester_user):
        if False:
            i = 10
            return i + 15
        from_model_kwargs = {'mask_secrets': True}
        rule = self._get_one(ref_or_id, permission_type=PermissionType.RULE_VIEW, requester_user=requester_user, from_model_kwargs=from_model_kwargs)
        result = self._append_view_properties([rule.json])[0]
        rule.json = result
        return rule

    def _append_view_properties(self, rules):
        if False:
            return 10
        (action_by_refs, trigger_by_refs, trigger_type_by_refs) = self._get_referenced_models(rules)
        for rule in rules:
            action_ref = rule.get('action', {}).get('ref', None)
            trigger_ref = rule.get('trigger', {}).get('ref', None)
            trigger_type_ref = rule.get('trigger', {}).get('type', None)
            action_db = action_by_refs.get(action_ref, None)
            if 'action' in rule:
                rule['action']['description'] = action_db.description if action_db else ''
            if 'trigger' in rule:
                rule['trigger']['description'] = ''
            trigger_db = trigger_by_refs.get(trigger_ref, None)
            if trigger_db:
                rule['trigger']['description'] = trigger_db.description
            if 'trigger' in rule and (not rule['trigger']['description']):
                trigger_type_db = trigger_type_by_refs.get(trigger_type_ref, None)
                if trigger_type_db:
                    rule['trigger']['description'] = trigger_type_db.description
        return rules

    def _get_referenced_models(self, rules):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reduces the number of queries to be made to the DB by creating sets of Actions, Triggers\n        and TriggerTypes.\n        '
        action_refs = set()
        trigger_refs = set()
        trigger_type_refs = set()
        for rule in rules:
            action_ref = rule.get('action', {}).get('ref', None)
            trigger_ref = rule.get('trigger', {}).get('ref', None)
            trigger_type_ref = rule.get('trigger', {}).get('type', None)
            if action_ref:
                action_refs.add(action_ref)
            if trigger_ref:
                trigger_refs.add(trigger_ref)
            if trigger_type_ref:
                trigger_type_refs.add(trigger_type_ref)
        action_by_refs = {}
        trigger_by_refs = {}
        trigger_type_by_refs = {}

        def ref_query_args(ref):
            if False:
                i = 10
                return i + 15
            return {'ref': ref}

        def name_pack_query_args(ref):
            if False:
                i = 10
                return i + 15
            resource_ref = ResourceReference.from_string_reference(ref=ref)
            return {'name': resource_ref.name, 'pack': resource_ref.pack}
        action_dbs = self._get_entities(model_persistence=Action, refs=action_refs, query_args=ref_query_args)
        for action_db in action_dbs:
            action_by_refs[action_db.ref] = action_db
        trigger_dbs = self._get_entities(model_persistence=Trigger, refs=trigger_refs, query_args=name_pack_query_args)
        for trigger_db in trigger_dbs:
            trigger_by_refs[trigger_db.get_reference().ref] = trigger_db
        trigger_type_dbs = self._get_entities(model_persistence=TriggerType, refs=trigger_type_refs, query_args=name_pack_query_args)
        for trigger_type_db in trigger_type_dbs:
            trigger_type_by_refs[trigger_type_db.get_reference().ref] = trigger_type_db
        return (action_by_refs, trigger_by_refs, trigger_type_by_refs)

    def _get_entities(self, model_persistence, refs, query_args):
        if False:
            while True:
                i = 10
        '\n        Returns all the entities for the supplied refs. model_persistence is the persistence\n        object that will be used to get to the correct query method and the query_args function\n        to return the ref specific query argument.\n\n        This is such a weirdly specific method that it is likely better only in this context.\n        '
        q = None
        for ref in refs:
            if not q:
                q = Q(**query_args(ref))
            else:
                q |= Q(**query_args(ref))
        if q:
            return model_persistence._get_impl().model.objects(q)
        return []
rule_view_controller = RuleViewController()