from tribler.core.components.knowledge.rules.rules_general_tags import general_rules
from tribler.core.components.knowledge.rules.tag_rules_base import extract_only_valid_tags

def test_extract_only_valid_tags():
    if False:
        for i in range(10):
            print('nop')
    assert set(extract_only_valid_tags('[valid-tag, i n v a l i d]', rules=general_rules)) == {'valid-tag'}