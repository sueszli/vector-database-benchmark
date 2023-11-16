from google.cloud import contact_center_insights_v1

def create_phrase_matcher_all_of(project_id: str) -> contact_center_insights_v1.PhraseMatcher:
    if False:
        i = 10
        return i + 15
    "Creates a phrase matcher that matches all specified queries.\n\n    Args:\n        project_id:\n            The project identifier. For example, 'my-project'.\n\n    Returns:\n        A phrase matcher.\n    "
    parent = contact_center_insights_v1.ContactCenterInsightsClient.common_location_path(project_id, 'us-central1')
    phrase_matcher = contact_center_insights_v1.PhraseMatcher()
    phrase_matcher.display_name = 'NON_SHIPPING_PHONE_SERVICE'
    phrase_matcher.type_ = contact_center_insights_v1.PhraseMatcher.PhraseMatcherType.ALL_OF
    phrase_matcher.active = True
    rule_group_phone_or_cellphone = contact_center_insights_v1.PhraseMatchRuleGroup()
    rule_group_phone_or_cellphone.type_ = contact_center_insights_v1.PhraseMatchRuleGroup.PhraseMatchRuleGroupType.ANY_OF
    for word in ['PHONE', 'CELLPHONE']:
        rule = contact_center_insights_v1.PhraseMatchRule()
        rule.query = word
        rule.config.exact_match_config = contact_center_insights_v1.ExactMatchConfig()
        rule_group_phone_or_cellphone.phrase_match_rules.append(rule)
    phrase_matcher.phrase_match_rule_groups.append(rule_group_phone_or_cellphone)
    rule_group_not_shipping_or_delivery = contact_center_insights_v1.PhraseMatchRuleGroup()
    rule_group_not_shipping_or_delivery.type_ = contact_center_insights_v1.PhraseMatchRuleGroup.PhraseMatchRuleGroupType.ALL_OF
    for word in ['SHIPPING', 'DELIVERY']:
        rule = contact_center_insights_v1.PhraseMatchRule()
        rule.query = word
        rule.negated = True
        rule.config.exact_match_config = contact_center_insights_v1.ExactMatchConfig()
        rule_group_not_shipping_or_delivery.phrase_match_rules.append(rule)
    phrase_matcher.phrase_match_rule_groups.append(rule_group_not_shipping_or_delivery)
    insights_client = contact_center_insights_v1.ContactCenterInsightsClient()
    phrase_matcher = insights_client.create_phrase_matcher(parent=parent, phrase_matcher=phrase_matcher)
    print(f'Created {phrase_matcher.name}')
    return phrase_matcher