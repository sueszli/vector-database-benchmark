import os
import sys
import requests
project = 'networkx'
core = 'core-developers'
emeritus = 'emeritus-developers'
steering = 'steering-council'
core_url = f'https://api.github.com/orgs/{project}/teams/{core}/members'
emeritus_url = f'https://api.github.com/orgs/{project}/teams/{emeritus}/members'
steering_url = f'https://api.github.com/orgs/{project}/teams/{steering}/members'
token = os.environ.get('GH_TOKEN', None)
if token is None:
    print('No token found.  Please export a GH_TOKEN with permissions to read team members.')
    sys.exit(-1)

def api(url):
    if False:
        i = 10
        return i + 15
    json = requests.get(url=url, headers={'Authorization': f'token {token}'}).json()
    if 'message' in json and json['message'] == 'Bad credentials':
        raise RuntimeError('Invalid token provided')
    else:
        return json
resp = api(core_url)
core = sorted(resp, key=lambda user: user['login'].lower())
resp = api(emeritus_url)
emeritus = sorted(resp, key=lambda user: user['login'].lower())
resp = api(steering_url)
steering = sorted(resp, key=lambda user: user['login'].lower())

def render_team(team):
    if False:
        while True:
            i = 10
    for member in team:
        profile = api(member['url'])
        print(f'''\n.. raw:: html\n\n   <div class="team-member">\n     <a href="https://github.com/{member['login']}" class="team-member-name">\n        <div class="team-member-photo">\n           <img\n             src="{member['avatar_url']}&s=40"\n             loading="lazy"\n             alt="Avatar picture of @{profile['login']}"\n           />\n        </div>\n        {(profile['name'] if profile['name'] else '@' + profile['login'])}\n     </a>\n     <div class="team-member-handle">@{member['login']}</div>\n   </div>\n''')
print('\n.. _core-developers-team:\n\nCore Developers\n---------------\n\nNetworkX development is guided by the following core team:\n\n')
render_team(core)
print('\n\nEmeritus Developers\n-------------------\n\nWe thank these previously-active core developers for their contributions to NetworkX.\n\n')
render_team(emeritus)
print('\n.. _steering-council-team:\n\nSteering Council\n----------------\n\n\n')
render_team(steering)