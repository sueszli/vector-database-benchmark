"""

    walle-web

    :copyright: © 2015-2017 walle-web.io
    :created time: 2017-03-25 11:15:01
    :author: wushuiyong@walle-web.io
"""
from flask import request, abort
from walle.api.api import SecurityResource
from walle.service.deployer import Deployer
from walle.service.extensions import permission

class RepoAPI(SecurityResource):
    actions = ['tags', 'branches', 'commits']

    @permission.upper_reporter
    def get(self, action, commit=None):
        if False:
            while True:
                i = 10
        '\n        fetch project list or one item\n        /project/<int:project_id>\n\n        :return:\n        '
        super(RepoAPI, self).get()
        project_id = request.args.get('project_id', '')
        if action in self.actions:
            self_action = getattr(self, action.lower(), None)
            return self_action(project_id=project_id)
        else:
            abort(404)

    def tags(self, project_id=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        fetch project list or one item\n        /tag/\n\n        :return:\n        '
        wi = Deployer(project_id=project_id)
        tag_list = wi.list_tag()
        tags = tag_list.stdout.strip().split('\n')
        return self.render_json(data={'tags': tags})

    def branches(self, project_id=None):
        if False:
            while True:
                i = 10
        '\n        fetch project list or one item\n        /tag/\n\n        :return:\n        '
        wi = Deployer(project_id=project_id)
        branches = wi.list_branch()
        return self.render_json(data={'branches': branches})

    def commits(self, project_id):
        if False:
            return 10
        '\n        fetch project list or one item\n        /tag/\n\n        :return:\n        '
        branch = request.args.get('branch', '')
        wi = Deployer(project_id=project_id)
        commits = wi.list_commit(branch)
        return self.render_json(data={'branches': commits})