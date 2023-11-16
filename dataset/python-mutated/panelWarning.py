import os, sys, json, time, public

class panelWarning:
    __path = '/www/server/panel/data/warning'
    __ignore = __path + '/ignore'
    __result = __path + '/result'

    def __init__(self):
        if False:
            print('Hello World!')
        if not os.path.exists(self.__ignore):
            os.makedirs(self.__ignore, 384)
        if not os.path.exists(self.__result):
            os.makedirs(self.__result, 384)

    def get_list(self, args):
        if False:
            print('Hello World!')
        p = public.get_modules('class/safe_warning')
        data = {'security': [], 'risk': [], 'ignore': []}
        for m_name in p.__dict__.keys():
            if p[m_name]._level == 0:
                continue
            m_info = {'title': p[m_name]._title, 'm_name': m_name, 'ps': p[m_name]._ps, 'version': p[m_name]._version, 'level': p[m_name]._level, 'ignore': p[m_name]._ignore, 'date': p[m_name]._date, 'tips': p[m_name]._tips, 'help': p[m_name]._help}
            result_file = self.__result + '/' + m_name + '.pl'
            not_force = True
            if 'force' in args:
                not_force = m_info['ignore']
            if os.path.exists(result_file) and not_force:
                try:
                    (m_info['status'], m_info['msg'], m_info['check_time'], m_info['taking']) = json.loads(public.readFile(result_file))
                except:
                    if os.path.exists(result_file):
                        os.remove(result_file)
                    continue
            else:
                try:
                    s_time = time.time()
                    (m_info['status'], m_info['msg']) = p[m_name].check_run()
                    m_info['taking'] = round(time.time() - s_time, 6)
                    m_info['check_time'] = int(time.time())
                    public.writeFile(result_file, json.dumps([m_info['status'], m_info['msg'], m_info['check_time'], m_info['taking']]))
                except:
                    continue
            if m_info['ignore']:
                data['ignore'].append(m_info)
            elif m_info['status']:
                data['security'].append(m_info)
            else:
                data['risk'].append(m_info)
        data['risk'] = sorted(data['risk'], key=lambda x: x['level'], reverse=True)
        data['security'] = sorted(data['security'], key=lambda x: x['level'], reverse=True)
        data['ignore'] = sorted(data['ignore'], key=lambda x: x['level'], reverse=True)
        return data

    def sync_rule(self):
        if False:
            return 10
        '\n            @name 从云端同步规则\n            @author hwliang<2020-08-05>\n            @return void\n        '

    def set_ignore(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n            @name 设置指定项忽略状态\n            @author hwliang<2020-08-04>\n            @param dict_obj {\n                m_name<string> 模块名称\n            }\n            @return dict\n        '
        m_name = args.m_name.strip()
        ignore_file = self.__ignore + '/' + m_name + '.pl'
        if os.path.exists(ignore_file):
            os.remove(ignore_file)
        else:
            public.writeFile(ignore_file, '1')
        return public.returnMsg(True, '设置成功!')

    def check_find(self, args):
        if False:
            i = 10
            return i + 15
        '\n            @name 检测指定项\n            @author hwliang<2020-08-04>\n            @param dict_obj {\n                m_name<string> 模块名称\n            }\n            @return dict\n        '
        try:
            m_name = args.m_name.strip()
            p = public.get_modules('class/safe_warning')
            m_info = {'title': p[m_name]._title, 'm_name': m_name, 'ps': p[m_name]._ps, 'version': p[m_name]._version, 'level': p[m_name]._level, 'ignore': p[m_name]._ignore, 'date': p[m_name]._date, 'tips': p[m_name]._tips, 'help': p[m_name]._help}
            result_file = self.__result + '/' + m_name + '.pl'
            s_time = time.time()
            (m_info['status'], m_info['msg']) = p[m_name].check_run()
            m_info['taking'] = round(time.time() - s_time, 4)
            m_info['check_time'] = int(time.time())
            public.writeFile(result_file, json.dumps([m_info['status'], m_info['msg'], m_info['check_time'], m_info['taking']]))
            return public.returnMsg(True, '已重新检测')
        except:
            return public.returnMsg(False, '错误的模块名称')