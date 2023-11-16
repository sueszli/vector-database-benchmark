def assert_that(value, matcher):
    if False:
        print('Hello World!')
    assert matcher.matches(value), matcher.describe_mismatch(value)