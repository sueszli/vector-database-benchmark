def test_assertion_error_not_raised_when_asserts_errors_is_false(graphql_client):
    if False:
        return 10
    query = '{  }'
    try:
        graphql_client.query(query, asserts_errors=False)
    except AssertionError:
        raise AssertionError