"""
.. module: security_monkey.watchers.github.org
    :platform: Unix
    :synopsis: Auditor for GitHub Organizations


.. version:: $$VERSION$$
.. moduleauthor:: Mike Grima <mgrima@netflix.com>

"""
from security_monkey.auditor import Auditor
from security_monkey.watchers.github.org import GitHubOrg

class GitHubOrgAuditor(Auditor):
    index = GitHubOrg.index
    i_am_singular = GitHubOrg.i_am_singular
    i_am_plural = GitHubOrg.i_am_plural

    def __init__(self, accounts=None, debug=False):
        if False:
            return 10
        super(GitHubOrgAuditor, self).__init__(accounts=accounts, debug=debug)

    def check_for_public_repo(self, org_item):
        if False:
            for i in range(10):
                print('nop')
        '\n        Organizational view that it has public repositories. Default score of 0. This is mostly\n        informational.\n        :param org_item:\n        :return:\n        '
        tag = 'Organization contains public repositories.'
        if org_item.config['public_repos'] > 0:
            self.add_issue(0, tag, org_item, notes='Organization contains public repositories')

    def check_for_non_twofa_members(self, org_item):
        if False:
            return 10
        "\n        Alert if the org has users that don't have 2FA enabled.\n\n        Will keep this at a level of 2 -- unles there are admins without 2FA, then that is level 10!\n        :param org_item:\n        :return:\n        "
        tag = 'Organization contains users without 2FA enabled.'
        owner_no_twofa = 'Organization owner does NOT have 2FA enabled!'
        if len(org_item.config['no_2fa_members']) > 0:
            self.add_issue(2, tag, org_item, notes='Organization contains users without 2FA enabled')
            for notwofa in org_item.config['no_2fa_members']:
                if notwofa in org_item.config['owners']:
                    self.add_issue(10, owner_no_twofa, org_item, notes='Organization OWNER: {} does NOT have 2FA enabled!'.format(notwofa))