import os
import sys
from twisted.internet import defer
from buildbot.data import connector
from buildbot.data.graphql import GraphQLConnector
from buildbot.test.fake import fakemaster
from buildbot.util import in_reactor

@in_reactor
@defer.inlineCallbacks
def gengraphql(config):
    if False:
        return 10
    master = (yield fakemaster.make_master(None, wantRealReactor=True))
    data = connector.DataConnector()
    yield data.setServiceParent(master)
    graphql = GraphQLConnector()
    yield graphql.setServiceParent(master)
    graphql.data = data
    master.config.www = {'graphql': {'debug': True}}
    graphql.reconfigServiceWithBuildbotConfig(master.config)
    yield master.startService()
    if config['out'] != '--':
        dirs = os.path.dirname(config['out'])
        if dirs and (not os.path.exists(dirs)):
            os.makedirs(dirs)
        f = open(config['out'], 'w', encoding='utf-8')
    else:
        f = sys.stdout
    schema = graphql.get_schema()
    f.write(schema)
    f.close()
    yield master.stopService()
    return 0