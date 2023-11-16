import sys
import io
import os
import time
import json
sys.path.append(os.getcwd() + '/class/core')
import mw
app_debug = False
if mw.isAppleSystem():
    app_debug = True

def getPluginName():
    if False:
        return 10
    return 'rsyncd'

def getPluginDir():
    if False:
        while True:
            i = 10
    return mw.getPluginDir() + '/' + getPluginName()

def getServerDir():
    if False:
        return 10
    return mw.getServerDir() + '/' + getPluginName()

def getTaskConf():
    if False:
        for i in range(10):
            print('nop')
    conf = getServerDir() + '/task_config.json'
    return conf

def getConfigData():
    if False:
        i = 10
        return i + 15
    try:
        return json.loads(mw.readFile(getTaskConf()))
    except:
        pass
    return []

def getConfigTpl():
    if False:
        for i in range(10):
            print('nop')
    tpl = {'name': '', 'task_id': -1}
    return tpl

def createBgTask(data):
    if False:
        while True:
            i = 10
    removeBgTask()
    for d in data:
        if d['realtime'] == 'false':
            createBgTaskByName(d['name'], d)

def createBgTaskByName(name, args):
    if False:
        print('Hello World!')
    cfg = getConfigTpl()
    _name = '[勿删]同步插件定时任务[' + name + ']'
    res = mw.M('crontab').field('id, name').where('name=?', (_name,)).find()
    if res:
        return True
    if 'task_id' in cfg.keys() and cfg['task_id'] > 0:
        res = mw.M('crontab').field('id, name').where('id=?', (cfg['task_id'],)).find()
        if res and res['id'] == cfg['task_id']:
            print('计划任务已经存在!')
            return True
    import crontab_api
    api = crontab_api.crontab_api()
    period = args['period']
    _hour = ''
    _minute = ''
    _where1 = ''
    _type_day = 'day'
    if period == 'day':
        _type_day = 'day'
        _hour = args['hour']
        _minute = args['minute']
    elif period == 'minute-n':
        _type_day = 'minute-n'
        _where1 = args['minute-n']
        _minute = ''
    cmd = '\nrname=%s\nplugin_path=%s\nlogs_file=$plugin_path/send/${rname}/run.log\n' % (name, getServerDir())
    cmd += 'echo "★【`date +"%Y-%m-%d %H:%M:%S"`】 STSRT" >> $logs_file' + '\n'
    cmd += 'echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" >> $logs_file' + '\n'
    cmd += 'bash $plugin_path/send/${rname}/cmd >> $logs_file 2>&1' + '\n'
    cmd += 'echo "【`date +"%Y-%m-%d %H:%M:%S"`】 END★" >> $logs_file' + '\n'
    cmd += 'echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" >> $logs_file' + '\n'
    params = {'name': _name, 'type': _type_day, 'week': '', 'where1': _where1, 'hour': _hour, 'minute': _minute, 'save': '', 'backup_to': '', 'stype': 'toShell', 'sname': '', 'sbody': cmd, 'urladdress': ''}
    task_id = api.add(params)
    if task_id > 0:
        cfg['task_id'] = task_id
        cfg['name'] = name
        _dd = getConfigData()
        _dd.append(cfg)
        mw.writeFile(getTaskConf(), json.dumps(_dd))

def removeBgTask():
    if False:
        return 10
    cfg_list = getConfigData()
    for x in range(len(cfg_list)):
        cfg = cfg_list[x]
        if 'task_id' in cfg.keys() and cfg['task_id'] > 0:
            res = mw.M('crontab').field('id, name').where('id=?', (cfg['task_id'],)).find()
            if res and res['id'] == cfg['task_id']:
                import crontab_api
                api = crontab_api.crontab_api()
                data = api.delete(cfg['task_id'])
                if data[0]:
                    cfg['task_id'] = -1
                    cfg_list[x] = cfg
                    mw.writeFile(getTaskConf(), '[]')
                    return True
    return False
if __name__ == '__main__':
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == 'remove':
            removeBgTask()
        elif action == 'add':
            createBgTask()