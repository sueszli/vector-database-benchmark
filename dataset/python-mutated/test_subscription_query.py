import pytest
from django.core.exceptions import ValidationError
from graphql import GraphQLError
from graphql.error import GraphQLSyntaxError
from saleor.webhook.error_codes import WebhookErrorCode
from ..subscription_query import IsFragment, SubscriptionQuery

def test_subscription_query():
    if False:
        return 10
    query = '\n        fragment EventFragment on Event {\n          ... on OrderUpdated {\n            order {\n                id\n            }\n          }\n          ... on OrderCreated {\n            order {\n                id\n            }\n          }\n        }\n\n        subscription {\n          event {\n            ... EventFragment\n            ... on ProductCreated {\n                product {\n                    id\n                }\n            }\n          }\n        }\n    '
    subscription_query = SubscriptionQuery(query)
    assert subscription_query.is_valid
    assert subscription_query.ast
    assert not subscription_query.errors
    assert subscription_query.events == ['order_created', 'order_updated', 'product_created']

@pytest.mark.parametrize(('query', 'events'), [('\n            subscription {\n              event {\n                ...on OrderCreated {\n                  order {\n                    id\n                  }\n                }\n              }\n            }\n            ', ['order_created']), ('\n            fragment OrderFragment on Order {\n              id\n              number\n              lines {\n                id\n              }\n            }\n            subscription {\n              event {\n                ...on OrderCreated {\n                  order {\n                    ...OrderFragment\n                  }\n                }\n              }\n            }\n            ', ['order_created']), ('\n            fragment OrderFragment on Order {\n                id\n            }\n\n            fragment EventFragment on Event {\n              issuedAt\n              ... on OrderUpdated {\n                order {\n                    ... OrderFragment\n                }\n              }\n            }\n\n            subscription {\n              event {\n                ... EventFragment\n              }\n            }\n            ', ['order_updated']), ('\n            subscription {\n              event{\n                ... on OrderCreated{\n                  order{\n                    id\n                  }\n                }\n                ... on OrderFullyPaid{\n                  order{\n                    id\n                  }\n                }\n                ... on ProductCreated{\n                  product{\n                    id\n                  }\n                }\n              }\n            }\n            ', ['order_created', 'order_fully_paid', 'product_created']), ('\n            fragment MyFragment on Event {\n                ... on OrderCreated{\n                  order{\n                    id\n                  }\n                }\n                ... on OrderUpdated{\n                  order{\n                    id\n                  }\n                }\n                ... on ProductCreated{\n                  product{\n                    id\n                  }\n                }\n            }\n            subscription {\n                event {\n                    ... MyFragment\n                }\n            }\n            ', ['order_created', 'order_updated', 'product_created']), ('\n            fragment EventFragment on Event {\n              ... on OrderUpdated {\n                order {\n                    id\n                }\n              }\n              ... on OrderCreated {\n                order {\n                    id\n                }\n              }\n            }\n\n            subscription {\n              event {\n                ... EventFragment\n                ... on ProductCreated {\n                    product {\n                        id\n                    }\n                }\n              }\n            }\n            ', ['order_updated', 'order_created', 'product_created']), ('\n            subscription InvoiceRequested {\n              event {\n                ...InvoiceRequestedPayload\n                }\n              }\n              fragment InvoiceRequestedPayload on InvoiceRequested {\n                invoice {\n                  id\n                }\n              }\n            ', ['invoice_requested']), ('\n            subscription{\n              event{\n                ...on ProductUpdated{\n                  product{\n                    id\n                  }\n                }\n              }\n              event{\n                ...on ProductCreated{\n                  product{\n                    id\n                  }\n                }\n              }\n            }\n            ', ['product_updated', 'product_created'])])
def test_get_event_type_from_subscription(query, events):
    if False:
        return 10
    subscription_query = SubscriptionQuery(query)
    assert subscription_query.is_valid
    assert subscription_query.events == sorted(events)

@pytest.mark.parametrize(('query', 'error_msg', 'error_type', 'error_code'), [('\n            mutation SomeMutation {\n                someMutation(input: {}) {\n                    result {\n                        id\n                    }\n                }\n            }\n            ', 'Cannot query field "someMutation" on type "Mutation".', GraphQLError, WebhookErrorCode.GRAPHQL_ERROR), ('\n            subscription {\n                event {\n                    ... MyFragment\n                }\n            }\n            ', 'Unknown fragment "MyFragment".', GraphQLError, WebhookErrorCode.GRAPHQL_ERROR), ('\n            fragment NotUsedEvents on Order {\n              id\n            }\n            subscription {\n              event {\n                ... on OrderUpdated {\n                  order {\n                    id\n                  }\n                }\n              }\n            }\n            ', 'Fragment "NotUsedEvents" is never used.', GraphQLError, WebhookErrorCode.GRAPHQL_ERROR), ('\n            query {{\n            }\n            ', 'Syntax Error GraphQL (2:20) Expected Name, found', GraphQLSyntaxError, WebhookErrorCode.SYNTAX), ('\n            query {\n              channels {\n                name\n              }\n            }\n            ', "Subscription operation can't be found.", ValidationError, WebhookErrorCode.MISSING_SUBSCRIPTION), ('\n            subscription {\n              event {\n                issuedAt\n              }\n            }\n            ', "Can't find a single event.", ValidationError, WebhookErrorCode.MISSING_EVENT)])
def test_query_validation(query, error_msg, error_type, error_code):
    if False:
        i = 10
        return i + 15
    subscription_query = SubscriptionQuery(query)
    assert not subscription_query.is_valid
    error = subscription_query.errors[0]
    assert isinstance(error, error_type)
    assert error_msg in error.message
    assert error_code.value == subscription_query.error_code

def test_get_events_from_field():
    if False:
        i = 10
        return i + 15
    query = '\n        fragment EventFragment on Event {\n          ... on OrderUpdated {\n            order {\n                id\n            }\n          }\n        }\n        subscription {\n          event {\n            ... on OrderCreated {\n              order {\n                id\n              }\n            }\n            something\n            somethingElse\n            ... on OrderFullyPaid {\n              order {\n                id\n              }\n            }\n            ... EventFragment\n          }\n        }\n        '
    subscription_query = SubscriptionQuery(query)
    subscription = subscription_query._get_subscription(subscription_query.ast)
    event_fields = subscription_query._get_event_types_from_subscription(subscription)
    result = {}
    for event_field in event_fields:
        subscription_query._get_events_from_field(event_field, result)
    assert result == {'OrderCreated': IsFragment.FALSE, 'OrderFullyPaid': IsFragment.FALSE, 'EventFragment': IsFragment.TRUE}