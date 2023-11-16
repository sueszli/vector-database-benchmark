"""Python test mode enabler.

Enables test mode for tf.data.

The test mode can be used to set up custom values for features and
experiments as required in the unit tests.

For example, if `warm_start` feature needs to be enabled exclusively for the
unit tests, the tests can enable the test mode using `toggle_test_mode` and
the default value of `warm_start` can be set as per the value of `TEST_MODE`.
"""
TEST_MODE = False

def toggle_test_mode(test_mode):
    if False:
        print('Hello World!')
    global TEST_MODE
    TEST_MODE = test_mode