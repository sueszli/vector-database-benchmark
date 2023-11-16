import types
from unittest import mock
import pytest
from django.conf import settings
from sentry.utils import json, kafka_config, outcomes
from sentry.utils.outcomes import Outcome, track_outcome

@pytest.fixture(autouse=True)
def setup():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(kafka_config, 'get_kafka_producer_cluster_options') as mck_get_options:
        with mock.patch.object(outcomes, 'KafkaPublisher') as mck_publisher:
            with mock.patch.object(outcomes, 'outcomes_publisher', None):
                with mock.patch.object(outcomes, 'billing_publisher', None):
                    yield types.SimpleNamespace(mock_get_kafka_producer_cluster_options=mck_get_options, mock_publisher=mck_publisher)

@pytest.mark.parametrize('outcome, is_billing', [(Outcome.ACCEPTED, True), (Outcome.FILTERED, False), (Outcome.RATE_LIMITED, True), (Outcome.INVALID, False), (Outcome.ABUSE, False), (Outcome.CLIENT_DISCARD, False)])
def test_outcome_is_billing(outcome: Outcome, is_billing: bool):
    if False:
        print('Hello World!')
    '\n    Tests the complete behavior of ``is_billing``, used for routing outcomes to\n    different Kafka topics. This is more of a sanity check to prevent\n    unintentional changes.\n    '
    assert outcome.is_billing() is is_billing

@pytest.mark.parametrize('name, outcome', [('rate_limited', Outcome.RATE_LIMITED), ('RATE_LIMITED', Outcome.RATE_LIMITED)])
def test_parse_outcome(name, outcome):
    if False:
        print('Hello World!')
    '\n    Asserts *case insensitive* parsing of outcomes from their canonical names,\n    as used in the API and queries.\n    '
    assert Outcome.parse(name) == outcome

def test_track_outcome_default(setup):
    if False:
        for i in range(10):
            print('nop')
    '\n    Asserts an outcomes serialization roundtrip with defaults.\n\n    Additionally checks that non-billing outcomes are routed to the DEFAULT\n    outcomes cluster and topic, even if there is a separate cluster for billing\n    outcomes.\n    '
    with mock.patch.dict(settings.KAFKA_TOPICS, {settings.KAFKA_OUTCOMES_BILLING: {'cluster': 'different'}}):
        track_outcome(org_id=1, project_id=2, key_id=3, outcome=Outcome.INVALID, reason='project_id')
        (cluster_args, _) = setup.mock_get_kafka_producer_cluster_options.call_args
        assert cluster_args == (kafka_config.get_topic_definition(settings.KAFKA_OUTCOMES)['cluster'],)
        assert outcomes.outcomes_publisher
        ((topic_name, payload), _) = setup.mock_publisher.return_value.publish.call_args
        assert topic_name == settings.KAFKA_OUTCOMES
        data = json.loads(payload)
        del data['timestamp']
        assert data == {'org_id': 1, 'project_id': 2, 'key_id': 3, 'outcome': Outcome.INVALID.value, 'reason': 'project_id', 'event_id': None, 'category': None, 'quantity': 1}
        assert outcomes.billing_publisher is None

def test_track_outcome_billing(setup):
    if False:
        print('Hello World!')
    '\n    Checks that outcomes are routed to the SHARED topic within the same cluster\n    in default configuration.\n    '
    track_outcome(org_id=1, project_id=1, key_id=1, outcome=Outcome.ACCEPTED)
    (cluster_args, _) = setup.mock_get_kafka_producer_cluster_options.call_args
    assert cluster_args == (kafka_config.get_topic_definition(settings.KAFKA_OUTCOMES)['cluster'],)
    assert outcomes.outcomes_publisher
    ((topic_name, _), _) = setup.mock_publisher.return_value.publish.call_args
    assert topic_name == settings.KAFKA_OUTCOMES
    assert outcomes.billing_publisher is None

def test_track_outcome_billing_topic(setup):
    if False:
        print('Hello World!')
    '\n    Checks that outcomes are routed to the DEDICATED billing topic within the\n    same cluster in default configuration.\n    '
    with mock.patch.dict(settings.KAFKA_TOPICS, {settings.KAFKA_OUTCOMES_BILLING: {'cluster': kafka_config.get_topic_definition(settings.KAFKA_OUTCOMES)['cluster']}}):
        track_outcome(org_id=1, project_id=1, key_id=1, outcome=Outcome.ACCEPTED)
        (cluster_args, _) = setup.mock_get_kafka_producer_cluster_options.call_args
        assert cluster_args == (kafka_config.get_topic_definition(settings.KAFKA_OUTCOMES)['cluster'],)
        assert outcomes.outcomes_publisher
        ((topic_name, _), _) = setup.mock_publisher.return_value.publish.call_args
        assert topic_name == settings.KAFKA_OUTCOMES_BILLING
        assert outcomes.billing_publisher is None

def test_track_outcome_billing_cluster(settings, setup):
    if False:
        while True:
            i = 10
    '\n    Checks that outcomes are routed to the dedicated cluster and topic.\n    '
    with mock.patch.dict(settings.KAFKA_TOPICS, {settings.KAFKA_OUTCOMES_BILLING: {'cluster': 'different'}}):
        track_outcome(org_id=1, project_id=1, key_id=1, outcome=Outcome.ACCEPTED)
        (cluster_args, _) = setup.mock_get_kafka_producer_cluster_options.call_args
        assert cluster_args == ('different',)
        assert outcomes.billing_publisher
        ((topic_name, _), _) = setup.mock_publisher.return_value.publish.call_args
        assert topic_name == settings.KAFKA_OUTCOMES_BILLING
        assert outcomes.outcomes_publisher is None