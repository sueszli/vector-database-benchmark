"""
.. module: security_monkey.tests.auditors.github.test_team_auditor
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor::  Mike Grima <mgrima@netflix.com>

"""
from security_monkey.datastore import Account, AccountType, Technology
from security_monkey.tests import SecurityMonkeyTestCase
from security_monkey import db
from security_monkey.watchers.github.team import GitHubTeamItem
from security_monkey.auditors.github.team import GitHubTeamAuditor
CONFIG_ONE = {'id': 1, 'url': 'https://api.github.com/teams/1', 'name': 'Justice League', 'slug': 'justice-league', 'description': 'A great team.', 'privacy': 'secret', 'permission': 'pull', 'members_url': 'https://api.github.com/teams/1/members{/member}', 'repositories_url': 'https://api.github.com/teams/1/repos'}
CONFIG_TWO = {'id': 2, 'url': 'https://api.github.com/teams/2', 'name': 'Team2', 'slug': 'Team2', 'description': 'A great team.', 'privacy': 'closed', 'permission': 'admin', 'members_url': 'https://api.github.com/teams/2/members{/member}', 'repositories_url': 'https://api.github.com/teams/2/repos'}

class GitHubTeamAuditorTestCase(SecurityMonkeyTestCase):

    def pre_test_setup(self):
        if False:
            while True:
                i = 10
        self.gh_items = [GitHubTeamItem(account='Org-one', name='Org-one', arn='Org-one', config=CONFIG_ONE), GitHubTeamItem(account='Org-one', name='Org-one', arn='Org-one', config=CONFIG_TWO)]
        self.account_type = AccountType(name='GitHub')
        db.session.add(self.account_type)
        db.session.commit()
        db.session.add(Account(name='Org-one', account_type_id=self.account_type.id, identifier='Org-one', active=True, third_party=False))
        self.technology = Technology(name='team')
        db.session.add(self.technology)
        db.session.commit()

    def test_public_team_check(self):
        if False:
            for i in range(10):
                print('nop')
        team_auditor = GitHubTeamAuditor(accounts=['Org-one'])
        team_auditor.check_for_public_team(self.gh_items[0])
        team_auditor.check_for_public_team(self.gh_items[1])
        self.assertEqual(len(self.gh_items[1].audit_issues), 1)
        self.assertEqual(self.gh_items[1].audit_issues[0].score, 1)
        self.assertEqual(len(self.gh_items[0].audit_issues), 0)