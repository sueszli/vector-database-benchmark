"""
Manage PagerDuty users.

Example:

    .. code-block:: yaml

    ensure bruce test user 1:
        pagerduty.user_present:
            - name: 'Bruce TestUser1'
            - email: bruce+test1@lyft.com
            - requester_id: P1GV5NT

"""

def __virtual__():
    if False:
        return 10
    '\n    Only load if the pygerduty module is available in __salt__\n    '
    if 'pagerduty_util.get_resource' in __salt__:
        return 'pagerduty_user'
    return (False, 'pagerduty_util module could not be loaded')

def present(profile='pagerduty', subdomain=None, api_key=None, **kwargs):
    if False:
        return 10
    '\n    Ensure pagerduty user exists.\n    Arguments match those supported by\n    https://developer.pagerduty.com/documentation/rest/users/create.\n    '
    return __salt__['pagerduty_util.resource_present']('users', ['email', 'name', 'id'], None, profile, subdomain, api_key, **kwargs)

def absent(profile='pagerduty', subdomain=None, api_key=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Ensure pagerduty user does not exist.\n    Name can be pagerduty id, email address, or user name.\n    '
    return __salt__['pagerduty_util.resource_absent']('users', ['email', 'name', 'id'], profile, subdomain, api_key, **kwargs)