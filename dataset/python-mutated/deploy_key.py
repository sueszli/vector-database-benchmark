from subprocess import Popen, PIPE

def deploy_key(ip, ssh_pwd):
    if False:
        while True:
            i = 10
    cmd = '/usr/bin/expect /var/opt/adminset/main/lib/sshkey_deploy {} {}'.format(ip, ssh_pwd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    data = p.communicate()
    return data