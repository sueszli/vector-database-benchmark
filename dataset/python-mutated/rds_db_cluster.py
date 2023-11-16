"""
.. module: security_monkey.auditors.rds.rds_db_cluster
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Patrick Kelley <patrick@netflix.com>

"""
from security_monkey.auditor import Auditor
from security_monkey.watchers.rds.rds_db_cluster import RDSDBCluster
from security_monkey.watchers.security_group import SecurityGroup

class RDSDBClusterAuditor(Auditor):
    index = RDSDBCluster.index
    i_am_singular = RDSDBCluster.i_am_singular
    i_am_plural = RDSDBCluster.i_am_plural
    support_auditor_indexes = [SecurityGroup.index]

    def __init__(self, accounts=None, debug=False):
        if False:
            for i in range(10):
                print('nop')
        super(RDSDBClusterAuditor, self).__init__(accounts=accounts, debug=debug)

    def _get_listener_ports_and_protocols(self, item):
        if False:
            print('Hello World!')
        '\n        "port": 3306\n        '
        port = item.config.get('port')
        return dict(TCP=set([port]))

    def check_internet_accessible(self, item):
        if False:
            for i in range(10):
                print('nop')
        security_groups = item.config.get('vpc_security_groups', [])
        security_group_ids = {sg['VpcSecurityGroupId'] for sg in security_groups}
        sg_auditor_items = self.get_auditor_support_items(SecurityGroup.index, item.account)
        security_auditor_groups = [sg for sg in sg_auditor_items if sg.config.get('id') in security_group_ids]
        for sg in security_auditor_groups:
            for issue in sg.db_item.issues:
                if self._issue_matches_listeners(item, issue):
                    self.link_to_support_item_issues(item, sg.db_item, sub_issue_message=issue.issue, score=issue.score)