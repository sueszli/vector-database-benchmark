import os
import sys
from twisted.internet import defer
from buildbot import config as config_module
from buildbot.master import BuildMaster
from buildbot.scripts import base
from buildbot.util import in_reactor

@defer.inlineCallbacks
def doCleanupDatabase(config, master_cfg):
    if False:
        for i in range(10):
            print('nop')
    if not config['quiet']:
        print(f"cleaning database ({master_cfg.db['db_url']})")
    master = BuildMaster(config['basedir'])
    master.config = master_cfg
    db = master.db
    yield db.setup(check_version=False, verbose=not config['quiet'])
    res = (yield db.logs.getLogs())
    i = 0
    percent = 0
    saved = 0
    for log in res:
        saved += (yield db.logs.compressLog(log['id'], force=config['force']))
        i += 1
        if not config['quiet'] and percent != i * 100 / len(res):
            percent = i * 100 / len(res)
            print(f' {percent}%  {saved} saved')
            saved = 0
            sys.stdout.flush()
    if master_cfg.db['db_url'].startswith('sqlite'):
        if not config['quiet']:
            print('executing sqlite vacuum function...')

        def thd(engine):
            if False:
                return 10
            sqlite_conn = engine.connection.connection
            sqlite_conn.isolation_level = None
            sqlite_conn.execute('vacuum;').close()
        yield db.pool.do(thd)

@in_reactor
def cleanupDatabase(config):
    if False:
        return 10
    return _cleanupDatabase(config)

@defer.inlineCallbacks
def _cleanupDatabase(config):
    if False:
        i = 10
        return i + 15
    if not base.checkBasedir(config):
        return 1
    config['basedir'] = os.path.abspath(config['basedir'])
    orig_cwd = os.getcwd()
    try:
        os.chdir(config['basedir'])
        with base.captureErrors((SyntaxError, ImportError), f"Unable to load 'buildbot.tac' from '{config['basedir']}':"):
            configFile = base.getConfigFileFromTac(config['basedir'])
        with base.captureErrors(config_module.ConfigErrors, f"Unable to load '{configFile}' from '{config['basedir']}':"):
            master_cfg = base.loadConfig(config, configFile)
        if not master_cfg:
            return 1
        yield doCleanupDatabase(config, master_cfg)
        if not config['quiet']:
            print('cleanup complete')
    finally:
        os.chdir(orig_cwd)
    return 0