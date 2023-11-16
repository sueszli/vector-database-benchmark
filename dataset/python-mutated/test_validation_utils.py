import unittest2
from oslo_config import cfg
from st2auth.validation import validate_auth_backend_is_correctly_configured
from st2tests import config as tests_config
__all__ = ['ValidationUtilsTestCase']

class ValidationUtilsTestCase(unittest2.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(ValidationUtilsTestCase, self).setUp()
        tests_config.parse_args()

    def test_validate_auth_backend_is_correctly_configured_success(self):
        if False:
            i = 10
            return i + 15
        result = validate_auth_backend_is_correctly_configured()
        self.assertTrue(result)

    def test_validate_auth_backend_is_correctly_configured_invalid_backend(self):
        if False:
            return 10
        cfg.CONF.set_override(group='auth', name='mode', override='invalid')
        expected_msg = 'Invalid auth mode "invalid" specified in the config. Valid modes are: proxy, standalone'
        self.assertRaisesRegexp(ValueError, expected_msg, validate_auth_backend_is_correctly_configured)

    def test_validate_auth_backend_is_correctly_configured_backend_doesnt_expose_groups(self):
        if False:
            return 10
        cfg.CONF.set_override(group='auth', name='backend', override='flat_file')
        cfg.CONF.set_override(group='auth', name='backend_kwargs', override='{"file_path": "dummy"}')
        cfg.CONF.set_override(group='rbac', name='enable', override=True)
        cfg.CONF.set_override(group='rbac', name='sync_remote_groups', override=True)
        expected_msg = "Configured auth backend doesn't expose user group information. Disable remote group synchronization or"
        self.assertRaisesRegexp(ValueError, expected_msg, validate_auth_backend_is_correctly_configured)