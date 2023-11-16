import os
from flask import Flask
from flask import render_template
from buildbot.process.results import statusToString
mydashboardapp = Flask('test', root_path=os.path.dirname(__file__))
mydashboardapp.config['TEMPLATES_AUTO_RELOAD'] = True

@mydashboardapp.route('/index.html')
def main():
    if False:
        for i in range(10):
            print('nop')
    builders = mydashboardapp.buildbot_api.dataGet('/builders')
    builds = mydashboardapp.buildbot_api.dataGet('/builds', limit=20)
    for build in builds:
        build['properties'] = mydashboardapp.buildbot_api.dataGet(('builds', build['buildid'], 'properties'))
        build['results_text'] = statusToString(build['results'])
    graph_data = [{'x': 1, 'y': 100}, {'x': 2, 'y': 200}, {'x': 3, 'y': 300}, {'x': 4, 'y': 0}, {'x': 5, 'y': 100}, {'x': 6, 'y': 200}, {'x': 7, 'y': 300}, {'x': 8, 'y': 0}, {'x': 9, 'y': 100}, {'x': 10, 'y': 200}]
    return render_template('mydashboard.html', builders=builders, builds=builds, graph_data=graph_data)
c['www']['plugins']['wsgi_dashboards'] = [{'name': 'mydashboard', 'caption': 'My Dashboard', 'app': mydashboardapp, 'order': 5, 'icon': 'area-chart'}]