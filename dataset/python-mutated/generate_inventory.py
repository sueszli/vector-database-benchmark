import os
import time
import sys
import subprocess
import digitalocean
OLD = False
if os.environ.get('OLD', False):
    OLD = True
if OLD:
    TOKEN_FILE = '/srv/secrets-newsblur/keys/digital_ocean.readprod.token'
else:
    TOKEN_FILE = '/srv/secrets-newsblur/keys/digital_ocean.token'
try:
    api_token = open(TOKEN_FILE, 'r').read().strip()
except IOError:
    print(f' ---> Missing Digital Ocean API token: {TOKEN_FILE}')
    exit()
outfile = f"/srv/newsblur/ansible/inventories/digital_ocean{('.old' if OLD else '')}.ini"
ansible_inventory_cmd = f'do-ansible-inventory -t {api_token} --out {outfile}'
subprocess.call(ansible_inventory_cmd, shell=True)
with open(outfile, 'r') as original:
    data = original.read()
with open(outfile, 'w') as modified:
    modified.write('127.0.0.1 ansible_connection=local\n' + data)
exit()
do = digitalocean.Manager(token=api_token)
droplets = do.get_all_droplets()
print('\n ---> Checking droplets: %s\n' % ' '.join([d.name for d in droplets]))

def check_droplets_created():
    if False:
        for i in range(10):
            print('nop')
    i = 0
    droplets = do.get_all_droplets()
    for instance in droplets:
        if instance.status == 'new':
            print('.', end=' ')
            sys.stdout.flush()
            i += 1
            time.sleep(i)
            break
    else:
        print(' ---> All booted!')
        return True
i = 0
while True:
    if check_droplets_created():
        break