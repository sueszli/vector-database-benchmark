from . import Framework

class EnterpriseAdmin(Framework.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.enterprise = self.g.get_enterprise('beaver-group')

    def testAttributes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.enterprise.enterprise, 'beaver-group')
        self.assertEqual(self.enterprise.url, '/enterprises/beaver-group')
        self.assertEqual(repr(self.enterprise), 'Enterprise(enterprise="beaver-group")')

    def testGetConsumedLicenses(self):
        if False:
            while True:
                i = 10
        consumed_licenses = self.enterprise.get_consumed_licenses()
        self.assertEqual(consumed_licenses.total_seats_consumed, 102)
        self.assertEqual(consumed_licenses.total_seats_purchased, 103)

    def testGetEnterpriseUsers(self):
        if False:
            i = 10
            return i + 15
        enterprise_users = self.enterprise.get_consumed_licenses().get_users()
        enterprise_users_list = [[users.github_com_login, users.github_com_name, users.enterprise_server_user_ids, users.github_com_user, users.enterprise_server_user, users.visual_studio_subscription_user, users.license_type, users.github_com_profile, users.github_com_member_roles, users.github_com_enterprise_roles, users.github_com_verified_domain_emails, users.github_com_saml_name_id, users.github_com_orgs_with_pending_invites, users.github_com_two_factor_auth, users.enterprise_server_primary_emails, users.visual_studio_license_status, users.visual_studio_subscription_email, users.total_user_accounts] for users in enterprise_users]
        self.assertEqual(len(enterprise_users_list), 102)
        self.assertEqual(enterprise_users_list[42][0], 'beaver-user043')