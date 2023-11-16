import os
import time
import public

class log_analysis:
    path = '/www/server/panel/script/'
    log_analysis_path = '/www/server/panel/script/log_analysis.sh'

    def __init__(self):
        if False:
            print('Hello World!')
        if not os.path.exists(self.path + '/log/'):
            os.makedirs(self.path + '/log/')
        if not os.path.exists(self.log_analysis_path):
            log_analysis_data = 'help(){\n\techo  "Usage: ./action.sh [options] [FILE] [OUTFILE]     "\n\techo  "Options:"\n\techo  "xxx.sh san_log     [FILE] 获取成功访问请求中带有xss|sql|铭感信息|php代码执行 关键字的日志列表  [OUTFILE]   11"\n\techo  "xxx.sh san     [FILE] 获取成功访问请求中带有sql关键字的日志列表   [OUTFILE]   11  "  \n}\n\nif [ $# == 0 ]\nthen\n\thelp\n\texit\nfi\n\nif [ ! -e $2 ]\nthen\n\techo -e "$2: 日志文件不存在"\n\texit\nfi\n\nif [ ! -d "log" ]\nthen\n\tmkdir log\nfi\n\necho "[*] Starting ..."\n\nif  [ $1 == "san_log" ] \nthen\n    echo "1">./log/$3\n\techo "开始获取xss跨站脚本攻击日志..."\n\n\tgrep -E \' (200|302|301|500|444|403|304) \' $2  | grep -i -E "(javascript|data:|alert\\(|onerror=|%3Cimg%20src=x%20on.+=|%3Cscript|%3Csvg/|%3Ciframe/|%3Cscript%3E).*?HTTP/1.1" >./log/$3xss.log\n\n\techo "分析日志已经保存到./log/$3xss.log"\n\techo "扫描到攻击次数: "`cat ./log/$3xss.log |wc -l`\n\techo "20">./log/$3\n\n\n\techo  "开始获取sql注入漏洞攻击日志..." \n\techo "分析日志已经保存到./log/$3sql.log"\ngrep -E \' (200|302|301|500|444|403) \' $2 | grep -i -E "(from.+?information_schema.+|select.+(from|limit)|union(.*?)select|extractvalue\\(|case when|extractvalue\\(|updatexml\\(|sleep\\().*?HTTP/1.1" > ./log/$3sql.log\n    echo "扫描到攻击次数: "`cat ./log/$3sql.log |wc -l`\n    echo "40">./log/$3\n\n\techo -e "开始获取文件遍历/代码执行/扫描器信息/配置文件等相关日志"\n\tgrep -E \' (200|302|301|500|444|403) \' $2 | grep -i -E "(\\.\\.|WEB-INF|/etc|\\w\\{1,6\\}\\.jsp |\\w\\{1,6\\}\\.php|\\w+\\.xml |\\w+\\.log |\\w+\\.swp |\\w*\\.git |\\w*\\.svn |\\w+\\.json |\\w+\\.ini |\\w+\\.inc |\\w+\\.rar |\\w+\\.gz |\\w+\\.tgz|\\w+\\.bak |/resin-doc).*?HTTP/1.1" >./log/$3san.log\n\techo "分析日志已经保存到./log/$3san.log"\n\techo "扫描到攻击次数: "`cat ./log/$3san.log |wc -l`\n\techo "50">./log/$3\n\n\n\techo -e "开始获取php代码执行扫描日志"\n\tgrep -E \' (200|302|301|500|444|403) \' $2 | grep -i -E "(gopher://|php://|file://|phar://|dict://data://|eval\\(|file_get_contents\\(|phpinfo\\(|require_once\\(|copy\\(|\\_POST\\[|file_put_contents\\(|system\\(|base64_decode\\(|passthru\\(|\\/invokefunction\\&|=call_user_func_array).*?HTTP/1.1" >./log/$3php.log\n\techo "分析日志已经保存到./log/$3php.log"\n\techo "扫描到攻击次数: "`cat ./log/$3php.log |wc -l`\n\techo "60">./log/$3\n\n\n\techo -e "正在统计访问次数最多ip的次数和值"\n# \tcat $2|awk -F" " \'{print $1}\'|sort|uniq -c|sort -nrk 1 -t\' \'|head -100\n\tawk \'{print $1}\' $2 |sort|uniq -c |sort -nr |head -100 >./log/$3ip.log\n\techo "80">./log/$3\n\n\n    echo -e "正在统计访问次数最多的请求接口的url的次数和值"\n\tawk \'{print $7}\' $2 |sort|uniq -c |sort -nr |head -100 >./log/$3url.log\n\techo "100">./log/$3\n\n\nelif [ $1 == "san" ]\nthen\n    echo "1">./log/$3\n\techo "开始获取xss跨站脚本攻击日志..."\n\tgrep -E \' (200|302|301|500|444|403|304) \' $2  | grep -i -E "(javascript|data:|alert\\(|onerror=|%3Cimg%20src=x%20on.+=|%3Cscript|%3Csvg/|%3Ciframe/|%3Cscript%3E).*?HTTP/1.1" >./log/$3xss.log\n\techo "分析日志已经保存到./log/$3xss.log"\n\techo "扫描到攻击次数: "`cat ./log/$3xss.log |wc -l`\n\techo "20">./log/$3\n\n\techo  "开始获取sql注入漏洞攻击日志..." \n\techo "分析日志已经保存到./log/$3sql.log"\ngrep -E \' (200|302|301|500|444|403) \' $2 | grep -i -E "(from.+?information_schema.+|select.+(from|limit)|union(.*?)select|extractvalue\\(|case when|extractvalue\\(|updatexml\\(|sleep\\().*?HTTP/1.1" > ./log/$3sql.log\n    echo "扫描到攻击次数: "`cat ./log/$3sql.log |wc -l`\n    echo "40">./log/$3\n\n\techo -e "开始获取文件遍历/代码执行/扫描器信息/配置文件等相关日志"\n\tgrep -E \' (200|302|301|500|444|403) \' $2 | grep -i -E "(\\.\\.|WEB-INF|/etc|\\w\\{1,6\\}\\.jsp |\\w\\{1,6\\}\\.php|\\w+\\.xml |\\w+\\.log |\\w+\\.swp |\\w*\\.git |\\w*\\.svn |\\w+\\.json |\\w+\\.ini |\\w+\\.inc |\\w+\\.rar |\\w+\\.gz |\\w+\\.tgz|\\w+\\.bak |/resin-doc).*?HTTP/1.1" >./log/$3san.log\n\n\techo "分析日志已经保存到./log/$3san.log"\n\techo "扫描到攻击次数: "`cat ./log/$3san.log |wc -l`\n\techo "60">./log/$3\n\n\techo -e "开始获取php代码执行扫描日志"\n\tgrep -E \' (200|302|301|500|444|403) \' $2 | grep -i -E "(gopher://|php://|file://|phar://|dict://data://|eval\\(|file_get_contents\\(|phpinfo\\(|require_once\\(|copy\\(|\\_POST\\[|file_put_contents\\(|system\\(|base64_decode\\(|passthru\\(|\\/invokefunction\\&|=call_user_func_array).*?HTTP/1.1" >./log/$3php.log\n\techo "分析日志已经保存到./log/$3php.log"\n\techo "扫描到攻击次数: "`cat ./log/$3php.log |wc -l`\n\techo "100">./log/$3\n\nelse \n\thelp\nfi\n\necho "[*] shut down"\n'
            public.WriteFile(self.log_analysis_path, log_analysis_data)

    def get_log_format(self, path):
        if False:
            return 10
        '\n        @获取日志格式\n        '
        f = open(path, 'r')
        data = None
        for i in f:
            data = i.split()
            break
        f.close()
        if not data:
            return False
        if not public.check_ip(data[0]):
            return False
        if len(data) < 6:
            return False
        return True

    def log_analysis(self, get):
        if False:
            i = 10
            return i + 15
        '\n        分析日志\n        @param path:需要分析的日志\n        @return 返回具体的分析结果\n        @ 需要使用异步的方式进行扫描\n        '
        if not os.path.exists(get.path):
            return public.ReturnMsg(False, '没有该日志文件')
        if os.path.getsize(get.path) > 9433107294:
            return public.ReturnMsg(False, '日志文件太大！')
        if os.path.getsize(get.path) < 10:
            return public.ReturnMsg(False, '日志文件为空')
        log_path = public.Md5(get.path)
        if self.get_log_format(get.path):
            public.ExecShell('cd %s && bash %s san_log %s %s &' % (self.path, self.log_analysis_path, get.path, log_path))
        else:
            public.ExecShell('cd %s && bash %s san %s %s &' % (self.path, self.log_analysis_path, get.path, log_path))
        speed = self.path + '/log/' + log_path + '.time'
        public.WriteFile(speed, str(time.time()) + '[]' + time.strftime('%Y-%m-%d %X', time.localtime()) + '[]' + '0')
        return public.ReturnMsg(True, '启动扫描成功')

    def speed_log(self, get):
        if False:
            while True:
                i = 10
        '\n        扫描进度\n        @param path:扫描的日志文件\n        @return 返回进度\n        '
        path = get.path.strip()
        log_path = public.Md5(path)
        speed = self.path + '/log/' + log_path
        if os.path.getsize(speed) < 1:
            return public.ReturnMsg(False, '日志文件为空')
        if not os.path.exists(speed):
            return public.ReturnMsg(False, '该目录没有扫描')
        try:
            data = public.ReadFile(speed)
            data = int(data)
            if data == 100:
                (time_data, start_time, status) = public.ReadFile(self.path + '/log/' + log_path + '.time').split('[]')
                public.WriteFile(speed + '.time', str(time.time() - float(time_data)) + '[]' + start_time + '[]' + '1')
            return public.ReturnMsg(True, data)
        except:
            return public.ReturnMsg(True, 0)

    def get_log_count(self, path, is_body=False):
        if False:
            i = 10
            return i + 15
        count = 0
        if is_body:
            if not os.path.exists(path):
                return ''
            data = ''
            with open(path, 'r') as f:
                for i in f:
                    count += 1
                    data = data + i
                    if count >= 300:
                        break
            return data
        else:
            if not os.path.exists(path):
                return count
            with open(path, 'rb') as f:
                for i in f:
                    count += 1
            return count

    def get_result(self, get):
        if False:
            print('Hello World!')
        '\n        扫描结果\n        @param path:扫描的日志文件\n        @return 返回结果\n        '
        path = get.path.strip()
        log_path = public.Md5(path)
        speed = self.path + '/log/' + log_path
        result = {}
        if os.path.exists(speed):
            result['is_status'] = True
        else:
            result['is_status'] = False
        if os.path.exists(speed + '.time'):
            (time_data, start_time, status) = public.ReadFile(self.path + '/log/' + log_path + '.time').split('[]')
            if status == '1' or start_time == 1:
                result['time'] = time_data
                result['start_time'] = start_time
        else:
            result['time'] = '0'
            result['start_time'] = '2022/2/22 22:22:22'
        if 'time' not in result:
            result['time'] = '0'
            result['start_time'] = '2022/2/22 22:22:22'
        result['xss'] = self.get_log_count(speed + 'xss.log')
        result['sql'] = self.get_log_count(speed + 'sql.log')
        result['san'] = self.get_log_count(speed + 'san.log')
        result['php'] = self.get_log_count(speed + 'php.log')
        result['ip'] = self.get_log_count(speed + 'ip.log')
        result['url'] = self.get_log_count(speed + 'url.log')
        return result

    def get_detailed(self, get):
        if False:
            i = 10
            return i + 15
        path = get.path.strip()
        log_path = public.Md5(path)
        speed = self.path + '/log/' + log_path
        type_list = ['xss', 'sql', 'san', 'php', 'ip', 'url']
        if get.type not in type_list:
            return public.ReturnMsg(False, '类型不匹配')
        if not os.path.exists(speed + get.type + '.log'):
            return public.ReturnMsg(False, '记录不存在')
        return self.get_log_count(speed + get.type + '.log', is_body=True)