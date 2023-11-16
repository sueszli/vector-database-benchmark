import json
import re
import subprocess
import textwrap
from shutil import which
from prompt_toolkit import prompt
from prompt_toolkit import print_formatted_text, HTML
UBUNTU_DSVM_IMAGE = 'microsoft-dsvm:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest'
vm_options = dict(gpu=dict(size='Standard_NC6s_v3', family='NCSv3', cores=6), cpu=dict(size='Standard_DS3_v2', family='DSv2', cores=4))
account_list_cmd = 'az account list -o table'
sub_id_list_cmd = 'az account list --query []."id" -o tsv'
region_list_cmd = 'az account list-locations --query []."name" -o tsv'
silent_login_cmd = 'az login --query "[?n]|[0]"'
set_account_sub_cmd = 'az account set -s {}'
provision_rg_cmd = 'az group create --name {} --location {}'
provision_vm_cmd = 'az vm create --resource-group {} --name {} --size {} --image {} --admin-username {} --admin-password {} --authentication-type password'
vm_ip_cmd = 'az vm show -d --resource-group {}-rg --name {} --query "publicIps" -o json'
quota_cmd = "az vm list-usage --location {} --query [?contains(localName,'{}')].{{max:limit,current:currentValue}}"
install_repo_cmd = 'az vm run-command invoke -g {}-rg -n {} --command-id RunShellScript --scripts'
install_repo_script = '<<<EOF\nls\nEOF\n'
tmp = '<<<EOF\nrm -rf computervision\nconda remove -n cv --all\ngit clone https://www.github.com/microsoft/computervision\ncd computervision\nconda env create -f environment.yml\ntmux\njupyter notebook --port 8888\nEOF'

def is_installed(cli_app: str) -> bool:
    if False:
        print('Hello World!')
    'Check whether `name` is on PATH and marked as executable.'
    return which(cli_app) is not None

def validate_password(password: str) -> bool:
    if False:
        return 10
    ' Checks that the password is valid.\n\n    Args:\n        password: password string\n\n    Returns: True if valid, else false.\n    '
    if len(password) < 12 or len(password) > 123:
        print_formatted_text(HTML('<ansired>Input must be between 12 and 123 characters. Please try again.</ansired>'))
        return False
    if len([c for c in password if c.islower()]) <= 0 or len([c for c in password if c.isupper()]) <= 0:
        print_formatted_text(HTML('<ansired>Input must contain a upper and a lower case character. Please try again.</ansired>'))
        return False
    if len([c for c in password if c.isdigit()]) <= 0:
        print_formatted_text(HTML('<ansired>Input must contain a digit. Please try again.</ansired>'))
        return False
    if len(re.findall('[\\W_]', password)) <= 0:
        print_formatted_text(HTML('<ansired>Input must contain a special character. Please try again.</ansired>'))
        return False
    return True

def validate_vm_name(vm_name) -> bool:
    if False:
        while True:
            i = 10
    ' Checks that the vm name is valid.\n\n    Args:\n        vm_name: the name of the vm to check\n\n    Returns: True if valid, else false.\n    '
    if len(vm_name) < 1 or len(vm_name) > 64 - 3:
        print_formatted_text(HTML(f'<ansired>Input must be between 1 and {64 - 3} characters. Please try again.</ansired>'))
        return False
    if not bool(re.match('^[A-Za-z0-9-]*$', vm_name)):
        print_formatted_text(HTML('<ansired>You can only use alphanumeric characters and hyphens. Please try again.</ansired>'))
        return False
    return True

def check_valid_yes_no_response(input: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if input in ('Y', 'y', 'N', 'n'):
        return True
    else:
        print_formatted_text(HTML("<ansired>Enter 'y' or 'n'. Please try again.</ansired>"))
        return False

def yes_no_prompter(msg: str) -> bool:
    if False:
        return 10
    cond = None
    valid_response = False
    while not valid_response:
        cond = prompt(msg)
        valid_response = check_valid_yes_no_response(cond)
    return True if cond in ('Y', 'y') else False

def prompt_subscription_id() -> str:
    if False:
        while True:
            i = 10
    ' Prompt for subscription id. '
    subscription_id = None
    subscription_is_valid = False
    results = subprocess.run(sub_id_list_cmd.split(' '), stdout=subprocess.PIPE)
    subscription_ids = results.stdout.decode('utf-8').strip().split('\n')
    while not subscription_is_valid:
        subscription_id = prompt('Enter your subscription id (copy & paste it from above): ')
        if subscription_id in subscription_ids:
            subscription_is_valid = True
        else:
            print_formatted_text(HTML('<ansired>The subscription id you entered is not valid. Please try again.</ansired>'))
    return subscription_id

def prompt_vm_name() -> str:
    if False:
        for i in range(10):
            print('nop')
    ' Prompt for VM name. '
    vm_name = None
    vm_name_is_valid = False
    while not vm_name_is_valid:
        vm_name = prompt(f"Enter a name for your vm (ex. 'cv-datascience-vm'): ")
        vm_name_is_valid = validate_vm_name(vm_name)
    return vm_name

def prompt_region() -> str:
    if False:
        for i in range(10):
            print('nop')
    ' Prompt for region. '
    region = None
    region_is_valid = False
    results = subprocess.run(region_list_cmd.split(' '), stdout=subprocess.PIPE)
    valid_regions = results.stdout.decode('utf-8').strip().split('\n')
    while not region_is_valid:
        region = prompt(f"Enter a region for your vm (ex. 'eastus'): ")
        if region in valid_regions:
            region_is_valid = True
        else:
            print_formatted_text(HTML(textwrap.dedent('                        <ansired>The region you entered is invalid. You can run\n                        `az account list-locations` to see a list of the valid\n                        regions. Please try again.</ansired>                        ')))
    return region

def prompt_username() -> str:
    if False:
        while True:
            i = 10
    ' Prompt username. '
    username = None
    username_is_valid = False
    while not username_is_valid:
        username = prompt('Enter a username: ')
        if len(username) > 0:
            username_is_valid = True
        else:
            print_formatted_text(HTML('<ansired>Username cannot be empty. Please try again.</ansired>'))
    return username

def prompt_password() -> str:
    if False:
        return 10
    ' Prompt for password. '
    password = None
    password_is_valid = False
    while not password_is_valid:
        password = prompt('Enter a password: ', is_password=True)
        if not validate_password(password):
            continue
        password_match = prompt('Enter your password again: ', is_password=True)
        if password == password_match:
            password_is_valid = True
        else:
            print_formatted_text(HTML('<ansired>Your passwords do not match. Please try again.</ansired>'))
    return password

def prompt_use_gpu() -> str:
    if False:
        i = 10
        return i + 15
    ' Prompt for GPU or CPU. '
    return yes_no_prompter('Do you want to use a GPU-enabled VM (It will incur a higher cost) [y/n]: ')

def prompt_use_cpu_instead() -> str:
    if False:
        for i in range(10):
            print('nop')
    ' Prompt switch to using CPU. '
    return yes_no_prompter('Do you want to switch to using a CPU instead? (This will likely solve your out-of-quota problem) [y/n]: ')

def get_available_quota(region: str, vm_family: str) -> int:
    if False:
        while True:
            i = 10
    ' Get available quota of the subscription in the specified region.\n\n    Args:\n        region: the region to check\n        vm_family: the vm family to check\n\n    Returns: the available quota\n    '
    results = subprocess.run(quota_cmd.format(region, vm_family).split(' '), stdout=subprocess.PIPE)
    quota = json.loads(''.join(results.stdout.decode('utf-8')))
    return int(quota[0]['max']) - int(quota[0]['current'])

def print_intro_dialogue():
    if False:
        return 10
    print_formatted_text(HTML(textwrap.dedent('\n            Azure Data Science Virtual Machine Builder\n\n            This utility will help you create an Azure Data Science Ubuntu Virtual\n            Machine that you will be able to run your notebooks in. The VM will\n            be based on the Ubuntu DSVM image.\n\n            For more information about Ubuntu DSVMs, see here:\n            https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro\n\n            This utility will let you select a GPU machine or a CPU machine.\n\n            The GPU machine specs:\n                - size: Standard_NC6s_v3 (NVIDIA Tesla V100 GPUs)\n                - family: NC6s\n                - cores: 6\n\n            The CPU machine specs:\n                - size: Standard_DS3_v2 (Intel XeonÂ® E5-2673 v3 2.4 GHz (Haswell))\n                - family: DSv2\n                - cores: 4\n\n            Pricing information on the SKUs can be found here:\n            https://azure.microsoft.com/en-us/pricing/details/virtual-machines\n\n            To use this utility, you must have an Azure subscription which you can\n            get from azure.microsoft.com.\n\n            Answer the questions below to setup your machine.\n\n            ------------------------------------------\n            ')))

def check_az_cli_installed():
    if False:
        i = 10
        return i + 15
    if not is_installed('az'):
        print(textwrap.dedent('            You must have the Azure CLI installed. For more information on\n            installing the Azure CLI, see here:\n            https://docs.microsoft.com/en-us/cli/azure/?view=azure-cli-latest\n        '))
        exit(0)

def check_logged_in() -> bool:
    if False:
        return 10
    print('Checking to see if you are logged in...')
    results = subprocess.run(account_list_cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return False if 'az login' in str(results.stderr) else True

def log_in(logged_in: bool):
    if False:
        while True:
            i = 10
    if not logged_in:
        subprocess.run(silent_login_cmd.split(' '))
        print('\n')
    else:
        print_formatted_text(HTML("<ansigreen>Looks like you're already logged in.</ansigreen>\n"))

def show_accounts():
    if False:
        print('Hello World!')
    print('Here is a list of your subscriptions:')
    results = subprocess.run(account_list_cmd.split(' '), stdout=subprocess.PIPE)
    print_formatted_text(HTML(f"<ansigreen>{results.stdout.decode('utf-8')}</ansigreen>"))

def check_quota(region: str, vm: dict, subscription_id: str) -> dict:
    if False:
        for i in range(10):
            print('nop')
    if get_available_quota(region, vm['family']) < vm['cores']:
        print_formatted_text(HTML(textwrap.dedent(f"                <ansired>\n                The subscription '{subscription_id}' does not have enough\n                cores of {vm['family']} in the region: {region}.\n\n                To request more cores:\n                https://docs.microsoft.com/en-us/azure/azure-supportability/resource-manager-core-quotas-request\n\n                (If you selected GPU, you may try using CPU instead.)\n                </ansired>                ")))
        if prompt_use_cpu_instead():
            vm = vm_options['cpu']
        else:
            print_formatted_text(HTML('Exiting..'))
            exit()
    return vm

def create_rg(vm_name: str, region: str):
    if False:
        print('Hello World!')
    print_formatted_text(HTML('\n<ansiyellow>Creating the resource group.</ansiyellow>'))
    results = subprocess.run(provision_rg_cmd.format(f'{vm_name}-rg', region).split(' '), stdout=subprocess.PIPE)
    if 'Succeeded' in results.stdout.decode('utf-8'):
        print_formatted_text(HTML('<ansigreen>Your resource group was successfully created.</ansigreen>\n'))

def create_vm(vm_name: str, vm: dict, username: str, password: str):
    if False:
        print('Hello World!')
    print_formatted_text(HTML('<ansiyellow>Creating the Data Science VM. This may take up a few minutes...</ansiyellow>'))
    subprocess.run(provision_vm_cmd.format(f'{vm_name}-rg', vm_name, vm['size'], UBUNTU_DSVM_IMAGE, username, password).split(' '), stdout=subprocess.PIPE)

def get_vm_ip(vm_name: str) -> str:
    if False:
        i = 10
        return i + 15
    results = subprocess.run(vm_ip_cmd.format(vm_name, vm_name).split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vm_ip = results.stdout.decode('utf-8').strip().strip('"')
    if len(vm_ip) > 0:
        print_formatted_text(HTML('<ansigreen>VM creation succeeded.</ansigreen>\n'))
    return vm_ip

def install_repo(username: str, password: str, vm_ip: str, vm_name: str):
    if False:
        for i in range(10):
            print('nop')
    print_formatted_text(HTML('<ansiyellow>Setting up your machine...</ansiyellow>'))
    invoke_cmd = install_repo_cmd.format(vm_name, vm_name)
    cmds = invoke_cmd.split(' ')
    cmds.append(f"""<<<EOF\nexport PATH=/anaconda/bin:$PATH\nconda remove -n cv --all\ncd /home/{username}\nrm -rf computervision\ngit clone https://www.github.com/microsoft/computervision\nchmod 777 computervision\ncd computervision\nconda env create -f environment.yml\nsource activate cv\npython -m ipykernel install --user --name cv --display-name "Python (cv)"\njupyter notebook --port 8899 --allow-root --NotebookApp.token='' --NotebookApp.password='' &\nEOF""")
    subprocess.run(cmds, stdout=subprocess.PIPE)
    print_formatted_text(HTML('<ansigreen>Successfully installed the repo on the machine.</ansigreen>\n'))

def print_exit_dialogue(vm_name: str, vm_ip: str, region: str, username: str, subscription_id: str):
    if False:
        return 10
    print_formatted_text(HTML(textwrap.dedent(f'\n            DSVM creation is complete. We recommend saving the details below.\n            <ansiyellow>\n            VM information:\n                - vm_name:         {vm_name}\n                - ip:              {vm_ip}\n                - region:          {region}\n                - username:        {username}\n                - password:        ****\n                - resource_group:  {vm_name}-rg\n                - subscription_id: {subscription_id}\n            </ansiyellow>\n            To start/stop VM:\n            <ansiyellow>\n                $az vm stop -g {vm_name}-rg -n {vm_name}\n                $az vm start -g {vm_name}-rg -n {vm_name}\n            </ansiyellow>\n            To connect via ssh and tunnel:\n            <ansiyellow>\n                $ssh -L 8899:localhost:8899 {username}@{vm_ip}\n            </ansiyellow>\n            To delete the VM (this command is unrecoverable):\n            <ansiyellow>\n                $az group delete -n {vm_name}-rg\n            </ansiyellow>\n            Please remember that virtual machines will incur a cost on your\n            Azure subscription. Remember to stop your machine if you are not\n            using it to minimize the cost.            ')))

def vm_builder() -> None:
    if False:
        for i in range(10):
            print('nop')
    ' Interaction session to create a data science vm. '
    print_intro_dialogue()
    prompt('Press enter to continue...\n')
    check_az_cli_installed()
    logged_in = check_logged_in()
    log_in(logged_in)
    show_accounts()
    subscription_id = prompt_subscription_id()
    vm_name = prompt_vm_name()
    region = prompt_region()
    use_gpu = prompt_use_gpu()
    username = prompt_username()
    password = prompt_password()
    vm = vm_options['gpu'] if use_gpu else vm_options['cpu']
    vm = check_quota(region, vm, subscription_id)
    subprocess.run(set_account_sub_cmd.format(subscription_id).split(' '))
    create_rg(vm_name, region)
    create_vm(vm_name, vm, username, password)
    vm_ip = get_vm_ip(vm_name)
    install_repo(username, password, vm_ip, vm_name)
    print_exit_dialogue(vm_name, vm_ip, region, username, subscription_id)
if __name__ == '__main__':
    vm_builder()