from datetime import timedelta
from django.utils import timezone
from ....discount.models import Promotion
from ....product.models import ProductVariant
from ...utils import fetch_active_promotion_rules

def test_fetch_active_promotion_rules(promotion, product, channel_USD):
    if False:
        i = 10
        return i + 15
    variants = ProductVariant.objects.all()
    rules_per_promotion = fetch_active_promotion_rules(variants)
    rules_info = rules_per_promotion.get(promotion.id)
    assert rules_info
    assert len(rules_info) == 2
    rules_info.sort(key=lambda info: info.rule.name)
    (fixed_rule, percentage_rule) = promotion.rules.all().order_by('name')
    (fixed_rule_info, percentage_rule_info) = rules_info
    assert fixed_rule_info.rule == fixed_rule
    assert fixed_rule_info.variant_ids == []
    assert not fixed_rule_info.variants
    assert fixed_rule_info.channel_ids == [channel_USD.id]
    assert percentage_rule_info.rule == percentage_rule
    assert set(percentage_rule_info.variant_ids) == {variant.id for variant in variants}
    assert {variant for variant in percentage_rule_info.variants} == {variant for variant in variants}
    assert percentage_rule_info.channel_ids == [channel_USD.id]

def test_fetch_active_promotion_rules_from_different_promotions(promotion_list, product, collection):
    if False:
        for i in range(10):
            print('nop')
    Promotion.objects.update(start_date=timezone.now() - timedelta(days=1))
    variants = ProductVariant.objects.all()
    rules_per_promotion = fetch_active_promotion_rules(variants)
    assert len(rules_per_promotion) == len(promotion_list)

def test_fetch_active_promotion_rules_no_active_rules(product):
    if False:
        i = 10
        return i + 15
    variants = ProductVariant.objects.all()
    rules_per_promotion = fetch_active_promotion_rules(variants)
    assert not rules_per_promotion

def test_fetch_active_promotion_rules_with_given_date(promotion, product, channel_USD):
    if False:
        return 10
    variants = ProductVariant.objects.all()
    rules_per_promotion = fetch_active_promotion_rules(variants, date=timezone.now() + timedelta(days=1))
    rules_info = rules_per_promotion.get(promotion.id)
    assert rules_info
    assert len(rules_info) == 2
    rules_info.sort(key=lambda info: info.rule.name)
    (fixed_rule, percentage_rule) = promotion.rules.all().order_by('name')
    (fixed_rule_info, percentage_rule_info) = rules_info
    assert fixed_rule_info.rule == fixed_rule
    assert fixed_rule_info.variant_ids == []
    assert not fixed_rule_info.variants
    assert fixed_rule_info.channel_ids == [channel_USD.id]
    assert percentage_rule_info.rule == percentage_rule
    assert set(percentage_rule_info.variant_ids) == {variant.id for variant in variants}
    assert {variant for variant in percentage_rule_info.variants} == {variant for variant in variants}
    assert percentage_rule_info.channel_ids == [channel_USD.id]

def test_fetch_active_promotion_rules_no_matching_rules_for_given_date(promotion, product):
    if False:
        i = 10
        return i + 15
    promotion.start_date = timezone.now() + timedelta(days=5)
    promotion.save(update_fields=['start_date'])
    variants = ProductVariant.objects.all()
    rules_per_promotion = fetch_active_promotion_rules(variants, date=timezone.now() + timedelta(days=1))
    assert not rules_per_promotion