from google.cloud import contact_center_insights_v1

def create_phrase_matcher_any_of(project_id: str) -> contact_center_insights_v1.PhraseMatcher:
    if False:
        return 10
    "Creates a phrase matcher that matches any of the specified queries.\n\n    Args:\n        project_id:\n            The project identifier. For example, 'my-project'.\n\n    Returns:\n        A phrase matcher.\n    "
    parent = contact_center_insights_v1.ContactCenterInsightsClient.common_location_path(project_id, 'us-central1')
    phrase_matcher = contact_center_insights_v1.PhraseMatcher()
    phrase_matcher.display_name = 'PHONE_SERVICE'
    phrase_matcher.type_ = contact_center_insights_v1.PhraseMatcher.PhraseMatcherType.ANY_OF
    phrase_matcher.active = True
    rule_group = contact_center_insights_v1.PhraseMatchRuleGroup()
    rule_group.type_ = contact_center_insights_v1.PhraseMatchRuleGroup.PhraseMatchRuleGroupType.ANY_OF
    for word in ['PHONE', 'CELLPHONE']:
        rule = contact_center_insights_v1.PhraseMatchRule()
        rule.query = word
        rule.config.exact_match_config = contact_center_insights_v1.ExactMatchConfig()
        rule_group.phrase_match_rules.append(rule)
    phrase_matcher.phrase_match_rule_groups.append(rule_group)
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    phrase_matcher = insights_client.create_phrase_matcher(parent=parent, phrase_matcher=phrase_matcher)
    print(f'Created {phrase_matcher.name}')
    return phrase_matcher