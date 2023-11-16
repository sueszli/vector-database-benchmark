#!/usr/bin/env python3
#encoding=utf-8
import os
import sys
import time
import subprocess
import argparse
import uuid

from socket import *
from random import randint
from multiprocessing import Process

from mitmproxy.certs import CertStore
from mitmproxy.tools.main import mitmweb as web
from mitmproxy.options import CONF_DIR, CONF_BASENAME, KEY_SIZE
from mitmproxy.version import VERSION

from packaging.version import parse as ver

from lamda import __version__
from lamda.client import *


serial = None
cleaning = False
def cleanup(*args, **kwargs):
    global cleaning
    if cleaning is True:
        return
    cleaning = True
    log ("uninstall certificate")
    d.uninstall_ca_certificate(ca)
    log ("disable proxy")
    d.stop_gproxy()
    os._exit (0)


def add_server(command, spec):
    spec and command.append("--mode")
    spec and command.append(spec)


def log(*args):
    print (time.ctime(), *args)


def adb(*args):
    command = ["adb"]
    if serial is not None:
        command.extend(["-s", serial])
    command.extend(args)
    log (" ".join(command))
    proc = subprocess.Popen(command)
    return proc


def adb_tcp(action, aport, bport):
    p = adb(action, "tcp:{}".format(aport),
                    "tcp:{}".format(bport))
    return p


def reverse(aport, bport):
    return adb_tcp("reverse", aport, bport)


def forward(aport, bport):
    return adb_tcp("forward", aport, bport)


def get_default_interface_ip_imp(target):
    s = socket(AF_INET, SOCK_DGRAM)
    s.connect(( target, lamda ))
    return s.getsockname()[0]


def get_default_interface_ip(target):
    default = get_default_interface_ip_imp(target)
    ip = os.environ.get("LANIP", default)
    return ip


print (r"           __                 __            .__  __            ")
print (r"   _______/  |______ ________/  |_    _____ |__|/  |_  _____   ")
print (r"  /  ___/\   __\__  \\_  __ \   __\  /     \|  \   __\/     \  ")
print (r"  \___ \  |  |  / __ \|  | \/|  |   |  Y Y  \  ||  | |  Y Y  \ ")
print (r" /____  > |__| (____  /__|   |__|   |__|_|  /__||__| |__|_|  / ")
print (r"      \/            \/                    \/               \/  ")
print (r"                 Android HTTP Traffic Capture                  ")
print (r"%60s" %                ("lamda#v%s BY rev1si0n" % (__version__)))


pkgName = None
argp = argparse.ArgumentParser()

login = "lamda"
psw = uuid.uuid4().hex[::3]
cert = os.environ.get("CERTIFICATE")
proxy = int(os.environ.get("PROXYPORT",
                    randint(28080, 58080)))
webport = randint(28080, 58080)
lamda = int(os.environ.get("PORT",
                    65000))

def dnsopt(dns):
    return "reverse:dns://{}@{}".format(dns, proxy)
argp.add_argument("device", nargs=1)
argp.add_argument("-m", "--mode", default="regular")
argp.add_argument("--serial", type=str, default=None)
dns = argp.add_mutually_exclusive_group(required=False)
dns.add_argument("--dns", type=dnsopt, nargs="?",
                                    const="1.1.1.1")
dns.add_argument("--nameserver", type=str, default="")
args, extras = argp.parse_known_args()
serial = args.serial
host = args.device[0]

if ":" in host:
    host, pkgName = host.split(":")
if args.dns and ver(VERSION) < ver("9.0.0"):
    log ("dns mitm needs mitmproxy>=9.0.0")
    sys.exit (1)

server = get_default_interface_ip(host)
usb = server in ("127.0.0.1", "::1")

if cert:
    log ("ssl:", cert)
if usb and args.dns:
    log ("dns mitm not available over usb")
    sys.exit (1)
if usb and (forward(lamda, lamda).wait() != 0 or \
            reverse(proxy, proxy).wait() != 0):
    log ("forward failed")
    sys.exit (1)

# 创建设备实例
d = Device(host, port=lamda,
                 certificate=cert)

# 拼接证书文件路径
DIR = os.path.expanduser(CONF_DIR)
CertStore.from_store(DIR, CONF_BASENAME, KEY_SIZE)
ca = os.path.join(DIR, "mitmproxy-ca-cert.pem")

log ("install cacert: %s" % ca)
d.install_ca_certificate(ca)

# 初始化 proxy 配置
profile = GproxyProfile()
profile.type = GproxyType.HTTP_CONNECT
profile.nameserver = args.nameserver
if not usb and args.dns:
    profile.nameserver = "{}:{}".format(server, proxy)
profile.drop_udp = True

profile.host = server
profile.port = proxy

profile.login = login
profile.password = psw
log ("set proxy: %s:%s@%s:%s/%s" % (
                            login, psw,
                            server, proxy,
                            pkgName or "all"))
if pkgName is not None:
    profile.application.set(d.application(pkgName))
d.start_gproxy(profile)

command = []
# 设置 MITMPROXY 代理模式
add_server(command, args.mode)
add_server(command, args.dns)
command.append("--ssl-insecure")
# 代理认证，防止误绑定到公网被扫描
command.append("--proxyauth")
command.append("{}:{}".format(login, psw))
# 随机 web-port
command.append("--web-port")
command.append(str(webport))
command.append("--no-rawtcp")
command.append("--listen-port")
command.append(str(proxy))
# 追加额外传递的参数
command.extend(extras)

log (" ".join(command))

sys.exit = cleanup
log ("press CONTROL + C to stop")
proc = Process(target=web, name="mitmweb",
               args=(command,), daemon=True)
proc.run()
sys.exit(0)