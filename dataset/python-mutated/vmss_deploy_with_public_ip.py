import argparse
import subprocess

def get_ips(rg_name, vmss_name):
    if False:
        print('Hello World!')
    'Get public IPs of all VMs in VMSS\n    Args:\n        rg_name (str): Resource group name\n        vmss_name (str): VMSS name\n    '
    script = 'az vmss list-instance-public-ips --resource-group {rg} --name {vmss} | grep ipAddress'.format(rg=rg_name, vmss=vmss_name)
    run_script(script)

def run_script(script):
    if False:
        for i in range(10):
            print('nop')
    results = subprocess.run(script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if results.stderr is not None and len(results.stderr) > 0:
        raise RuntimeError(results.stderr.decode('utf-8'))
    print(results.stdout.decode('utf-8'))

def parse():
    if False:
        while True:
            i = 10
    'Parser'
    parser = argparse.ArgumentParser(description='Deploy VMSS with multiple user accounts')
    parser.add_argument('--name', type=str, help='Resource-group name to create')
    parser.add_argument('--location', type=str, help="Location to deploy resources (e.g. 'eastus')")
    parser.add_argument('--image', type=str, default='microsoft-dsvm:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest', help="VM image URN. Format='Publisher:Offer:Sku:Version'")
    parser.add_argument('--vm-sku', type=str, help="VM size (e.g. 'Standard_NC6s_v3')")
    parser.add_argument('--vm-count', type=int, help='Number of VMs to create')
    parser.add_argument('--admin-id', type=str, help='Admin user id for all VMs')
    parser.add_argument('--admin-pw', type=str, help='Admin user pw for all VMs')
    parser.add_argument('--post-script', type=str, help='Post deployment script to run on each VM')
    _args = parser.parse_args()
    if _args.name is None or _args.location is None:
        raise argparse.ArgumentError(None, '--name and --location should be provided')
    if _args.vm_sku is None or _args.vm_count is None:
        raise argparse.ArgumentError(None, '--vm-sku and --vm-count should be provided')
    if _args.admin_id is None or _args.admin_pw is None:
        raise argparse.ArgumentError(None, '--admin-id and --admin-pw should be provided')
    return _args
if __name__ == '__main__':
    args = parse()
    RG_NAME = args.name
    VMSS_NAME = '{}-vmss'.format(RG_NAME)
    print('Creating resource group...')
    create_rg = 'az group create --name {rg} --location {location}'.format(rg=RG_NAME, location=args.location)
    run_script(create_rg)
    print('\nDeploying VMSS...')
    create_vmss = 'az vmss create -g {rg} -n {vmss} --instance-count {vm_count} --image {image} --vm-sku {vm_sku} --public-ip-per-vm --admin-username {admin_id} --admin-password {admin_pw}'.format(rg=RG_NAME, vmss=VMSS_NAME, vm_count=args.vm_count, image=args.image, vm_sku=args.vm_sku, admin_id=args.admin_id, admin_pw=args.admin_pw)
    run_script(create_vmss)
    if args.post_script is not None:
        print('\nRun post-deployment script {}...'.format(args.post_script))
        run_post_script = 'az vmss list-instances -g {rg} -n {vmss} --query "[].id" --output tsv | az vmss run-command invoke --command-id RunShellScript --scripts @{post_script} --ids @-'.format(rg=RG_NAME, vmss=VMSS_NAME, post_script=args.post_script)
        run_script(run_post_script)
    print('\nVM instance ips:\n')
    get_ips(RG_NAME, VMSS_NAME)