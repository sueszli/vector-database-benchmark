import pytest
import salt.states.boto_secgroup as boto_secgroup
from salt.utils.odict import OrderedDict

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {boto_secgroup: {}}

def test__get_rule_changes_no_rules_no_change():
    if False:
        while True:
            i = 10
    '\n    tests a condition with no rules in present or desired group\n    '
    present_rules = []
    desired_rules = []
    assert boto_secgroup._get_rule_changes(desired_rules, present_rules) == ([], [])

def test__get_rule_changes_create_rules():
    if False:
        for i in range(10):
            print('nop')
    '\n    tests a condition where a rule must be created\n    '
    present_rules = [OrderedDict([('ip_protocol', 'tcp'), ('from_port', 22), ('to_port', 22), ('cidr_ip', '0.0.0.0/0')])]
    desired_rules = [OrderedDict([('ip_protocol', 'tcp'), ('from_port', 22), ('to_port', 22), ('cidr_ip', '0.0.0.0/0')]), OrderedDict([('ip_protocol', 'tcp'), ('from_port', 80), ('to_port', 80), ('cidr_ip', '0.0.0.0/0')])]
    rules_to_create = [OrderedDict([('ip_protocol', 'tcp'), ('from_port', 80), ('to_port', 80), ('cidr_ip', '0.0.0.0/0')])]
    assert boto_secgroup._get_rule_changes(desired_rules, present_rules) == ([], rules_to_create)

def test__get_rule_changes_delete_rules():
    if False:
        return 10
    '\n    tests a condition where a rule must be deleted\n    '
    present_rules = [OrderedDict([('ip_protocol', 'tcp'), ('from_port', 22), ('to_port', 22), ('cidr_ip', '0.0.0.0/0')]), OrderedDict([('ip_protocol', 'tcp'), ('from_port', 80), ('to_port', 80), ('cidr_ip', '0.0.0.0/0')])]
    desired_rules = [OrderedDict([('ip_protocol', 'tcp'), ('from_port', 22), ('to_port', 22), ('cidr_ip', '0.0.0.0/0')])]
    rules_to_delete = [OrderedDict([('ip_protocol', 'tcp'), ('from_port', 80), ('to_port', 80), ('cidr_ip', '0.0.0.0/0')])]
    assert boto_secgroup._get_rule_changes(desired_rules, present_rules) == (rules_to_delete, [])