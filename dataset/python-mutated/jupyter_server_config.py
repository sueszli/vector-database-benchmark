import os
from jupyterhub.singleuser.extension import JupyterHubAuthorizer

class GranularJupyterHubAuthorizer(JupyterHubAuthorizer):
    """Authorizer that looks for permissions in JupyterHub scopes"""

    def is_authorized(self, handler, user, action, resource):
        if False:
            for i in range(10):
                print('nop')
        filters = [f"!user={os.environ['JUPYTERHUB_USER']}", f"!server={os.environ['JUPYTERHUB_USER']}/{os.environ['JUPYTERHUB_SERVER_NAME']}"]
        required_scopes = set()
        for f in filters:
            required_scopes.update({f'custom:jupyter_server:{action}:{resource}{f}', f'custom:jupyter_server:{action}:*{f}'})
        have_scopes = self.hub_auth.check_scopes(required_scopes, user.hub_user)
        self.log.debug(f'{user.username} has permissions {have_scopes} required to {action} on {resource}')
        return bool(have_scopes)
c = get_config()
c.ServerApp.authorizer_class = GranularJupyterHubAuthorizer