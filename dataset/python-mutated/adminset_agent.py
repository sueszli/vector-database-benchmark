import os, re, platform, socket, time, json, threading
import psutil, schedule, requests
from subprocess import Popen, PIPE
import logging
AGENT_VERSION = '1.0'
token = 'HPcWR7l4NJNJ'
server_ip = '192.168.47.130'

def log(log_name, path=None):
    if False:
        for i in range(10):
            print('nop')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y%m%d %H:%M:%S', filename=path + log_name, filemode='ab+')
    return logging.basicConfig
log('agent.log', '/var/opt/adminset/client/')

def get_ip():
    if False:
        i = 10
        return i + 15
    try:
        hostname = socket.getfqdn(socket.gethostname())
        ipaddr = socket.gethostbyname(hostname)
    except Exception as msg:
        print(msg)
        ipaddr = ''
    return ipaddr

def get_dmi():
    if False:
        i = 10
        return i + 15
    p = Popen('dmidecode', stdout=PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    return stdout

def parser_dmi(dmidata):
    if False:
        for i in range(10):
            print('nop')
    pd = {}
    line_in = False
    for line in dmidata.split('\n'):
        if line.startswith('System Information'):
            line_in = True
            continue
        if line.startswith('\t') and line_in:
            (k, v) = [i.strip() for i in line.split(':')]
            pd[k] = v
        else:
            line_in = False
    return pd

def get_mem_total():
    if False:
        while True:
            i = 10
    cmd = 'grep MemTotal /proc/meminfo'
    p = Popen(cmd, stdout=PIPE, shell=True)
    data = p.communicate()[0]
    mem_total = data.split()[1]
    memtotal = int(round(int(mem_total) / 1024.0 / 1024.0, 0))
    return memtotal

def get_cpu_model():
    if False:
        return 10
    cmd = 'cat /proc/cpuinfo'
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    return stdout

def get_cpu_cores():
    if False:
        return 10
    cpu_cores = {'physical': psutil.cpu_count(logical=False) if psutil.cpu_count(logical=False) else 0, 'logical': psutil.cpu_count()}
    return cpu_cores

def parser_cpu(stdout):
    if False:
        i = 10
        return i + 15
    groups = [i for i in stdout.split('\n\n')]
    group = groups[-2]
    cpu_list = [i for i in group.split('\n')]
    cpu_info = {}
    for x in cpu_list:
        (k, v) = [i.strip() for i in x.split(':')]
        cpu_info[k] = v
    return cpu_info

def get_disk_info():
    if False:
        return 10
    ret = []
    cmd = "fdisk -l|egrep '^Disk\\s/dev/[a-z]+:\\s\\w*'"
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    for i in stdout.split('\n'):
        disk_info = i.split(',')
        if disk_info[0]:
            ret.append(disk_info[0])
    return ret

def post_data(url, data):
    if False:
        return 10
    try:
        r = requests.post(url, data)
        if r.text:
            logging.info(r.text)
        else:
            logging.info('Server return http status code: {0}'.format(r.status_code))
    except Exception as msg:
        logging.info(msg)
    return True

def asset_info():
    if False:
        i = 10
        return i + 15
    data_info = dict()
    data_info['memory'] = get_mem_total()
    data_info['disk'] = str(get_disk_info())
    cpuinfo = parser_cpu(get_cpu_model())
    cpucore = get_cpu_cores()
    data_info['cpu_num'] = cpucore['logical']
    data_info['cpu_physical'] = cpucore['physical']
    data_info['cpu_model'] = cpuinfo['model name']
    data_info['ip'] = get_ip()
    data_info['sn'] = parser_dmi(get_dmi())['Serial Number']
    data_info['vendor'] = parser_dmi(get_dmi())['Manufacturer']
    data_info['product'] = parser_dmi(get_dmi())['Version']
    data_info['osver'] = platform.linux_distribution()[0] + ' ' + platform.linux_distribution()[1] + ' ' + platform.machine()
    data_info['hostname'] = platform.node()
    data_info['token'] = token
    data_info['agent_version'] = AGENT_VERSION
    return json.dumps(data_info)

def asset_info_post():
    if False:
        return 10
    pversion = platform.python_version()
    pv = re.search('2.6', pversion)
    if not pv:
        osenv = os.environ['LANG']
        os.environ['LANG'] = 'us_EN.UTF8'
    logging.info('Get the hardwave infos from host:')
    logging.info(asset_info())
    logging.info('----------------------------------------------------------')
    post_data('http://{0}/cmdb/collect'.format(server_ip), asset_info())
    if not pv:
        os.environ['LANG'] = osenv
    return True

def get_sys_cpu():
    if False:
        for i in range(10):
            print('nop')
    sys_cpu = {}
    cpu_time = psutil.cpu_times_percent(interval=1)
    sys_cpu['percent'] = psutil.cpu_percent(interval=1)
    sys_cpu['lcpu_percent'] = psutil.cpu_percent(interval=1, percpu=True)
    sys_cpu['user'] = cpu_time.user
    sys_cpu['nice'] = cpu_time.nice
    sys_cpu['system'] = cpu_time.system
    sys_cpu['idle'] = cpu_time.idle
    sys_cpu['iowait'] = cpu_time.iowait
    sys_cpu['irq'] = cpu_time.irq
    sys_cpu['softirq'] = cpu_time.softirq
    sys_cpu['guest'] = cpu_time.guest
    return sys_cpu

def get_sys_mem():
    if False:
        i = 10
        return i + 15
    sys_mem = {}
    mem = psutil.virtual_memory()
    sys_mem['total'] = mem.total / 1024 / 1024
    sys_mem['percent'] = mem.percent
    sys_mem['available'] = mem.available / 1024 / 1024
    sys_mem['used'] = mem.used / 1024 / 1024
    sys_mem['free'] = mem.free / 1024 / 1024
    sys_mem['buffers'] = mem.buffers / 1024 / 1024
    sys_mem['cached'] = mem.cached / 1024 / 1024
    return sys_mem

def parser_sys_disk(mountpoint):
    if False:
        for i in range(10):
            print('nop')
    partitions_list = {}
    d = psutil.disk_usage(mountpoint)
    partitions_list['mountpoint'] = mountpoint
    partitions_list['total'] = round(d.total / 1024 / 1024 / 1024.0, 2)
    partitions_list['free'] = round(d.free / 1024 / 1024 / 1024.0, 2)
    partitions_list['used'] = round(d.used / 1024 / 1024 / 1024.0, 2)
    partitions_list['percent'] = d.percent
    return partitions_list

def get_sys_disk():
    if False:
        while True:
            i = 10
    sys_disk = {}
    partition_info = []
    partitions = psutil.disk_partitions()
    for p in partitions:
        partition_info.append(parser_sys_disk(p.mountpoint))
    sys_disk = partition_info
    return sys_disk

def get_nic():
    if False:
        print('Hello World!')
    key_info = psutil.net_io_counters(pernic=True).keys()
    recv = {}
    sent = {}
    for key in key_info:
        recv.setdefault(key, psutil.net_io_counters(pernic=True).get(key).bytes_recv)
        sent.setdefault(key, psutil.net_io_counters(pernic=True).get(key).bytes_sent)
    return (key_info, recv, sent)

def get_nic_rate(func):
    if False:
        for i in range(10):
            print('nop')
    (key_info, old_recv, old_sent) = func()
    time.sleep(1)
    (key_info, now_recv, now_sent) = func()
    net_in = {}
    net_out = {}
    for key in key_info:
        net_in.setdefault(key, (now_recv.get(key) - old_recv.get(key)) / 1024)
        net_out.setdefault(key, (now_sent.get(key) - old_sent.get(key)) / 1024)
    return (key_info, net_in, net_out)

def get_net_info():
    if False:
        print('Hello World!')
    net_info = []
    (key_info, net_in, net_out) = get_nic_rate(get_nic)
    for key in key_info:
        in_data = net_in.get(key)
        out_data = net_out.get(key)
        net_info.append({'nic_name': key, 'traffic_in': in_data, 'traffic_out': out_data})
    return net_info

def agg_sys_info():
    if False:
        while True:
            i = 10
    logging.info('Get the system infos from host:')
    sys_info = {'hostname': platform.node(), 'cpu': get_sys_cpu(), 'mem': get_sys_mem(), 'disk': get_sys_disk(), 'net': get_net_info(), 'token': token}
    logging.info(sys_info)
    json_data = json.dumps(sys_info)
    logging.info('----------------------------------------------------------')
    post_data('http://{0}/monitor/received/sys/info/'.format(server_ip), json_data)
    return True

def run_threaded(job_func):
    if False:
        i = 10
        return i + 15
    job_thread = threading.Thread(target=job_func)
    job_thread.start()

def get_pid():
    if False:
        print('Hello World!')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pid = str(os.getpid())
    with open(BASE_DIR + '/adminsetd.pid', 'wb+') as pid_file:
        pid_file.writelines(pid)

def clean_log():
    if False:
        for i in range(10):
            print('nop')
    os.system('> /var/opt/adminset/agent.log')
    logging.info('clean agent log')
if __name__ == '__main__':
    get_pid()
    asset_info_post()
    time.sleep(1)
    agg_sys_info()
    schedule.every(3600).seconds.do(run_threaded, asset_info_post)
    schedule.every(300).seconds.do(run_threaded, agg_sys_info)
    schedule.every().monday.at('00:20').do(run_threaded, clean_log)
    while True:
        schedule.run_pending()
        time.sleep(1)