import difflib
import os
import re
import sys
import yaml
root_path = sys.argv[4]

def read_yaml_ops():
    if False:
        print('Hello World!')
    ops_list = []
    yaml_path = root_path + '/paddle/phi/api/yaml/ops.yaml'
    legacy_yaml_path = root_path + '/paddle/phi/api/yaml/legacy_ops.yaml'
    with open(yaml_path, 'r') as f:
        ops_list = yaml.load(f, Loader=yaml.FullLoader)
    with open(legacy_yaml_path, 'r') as f:
        ops_list.extend(yaml.load(f, Loader=yaml.FullLoader))
    return ops_list

def read_api(api_file):
    if False:
        while True:
            i = 10
    with open(api_file, 'r') as f:
        pr_apis = f.read()
        pr_apis = pr_apis.splitlines()
    result = []
    for api in pr_apis:
        if api.find('args') == -1:
            continue
        result.append(api)
    return result

def get_api_args(api_item):
    if False:
        print('Hello World!')
    result = re.search('args=\\[(?P<args>[^\\]]*)\\]', api_item)
    result = [param.strip().replace("'", '') for param in result.group('args').split(',')]
    if result[-1] == 'name':
        result = result[:-1]
    return result

def get_api_name(api_item):
    if False:
        i = 10
        return i + 15
    if api_item[0] == '+' or api_item[0] == '-' or api_item[0] == ' ':
        return api_item.split(' ')[1].split('.')[-1]
    else:
        return api_item.split(' ')[0].split('.')[-1]

def get_yaml_op_args(op_args):
    if False:
        return 10
    args_list = op_args[1:-1].split(',')
    args_list = [args.split('=')[0].strip() for args in args_list]
    return [param.split(' ')[-1].strip() for param in args_list]

def get_api_diff(dev_api_file, pr_api_file):
    if False:
        print('Hello World!')
    develop_apis = read_api(dev_api_file)
    pr_apis = read_api(pr_api_file)
    differ = difflib.Differ()
    diff_obj = differ.compare(develop_apis, pr_apis)
    result = []
    for each_diff in diff_obj:
        result.append(each_diff)
    return result

def get_yaml_diff(branch):
    if False:
        print('Hello World!')
    ops_yaml_path = root_path + '/paddle/phi/api/yaml/ops.yaml'
    legacy_yaml_path = root_path + '/paddle/phi/api/yaml/legacy_ops.yaml'
    git_cmd = 'git diff -U0 upstream/' + branch + ' ' + ops_yaml_path + ' ' + legacy_yaml_path
    yaml_diff = os.popen(git_cmd).readlines()
    result = []
    for line in yaml_diff:
        result.append(line.strip('\r\n'))
    return result
api_diffs = get_api_diff(sys.argv[1], sys.argv[2])
yaml_diffs = get_yaml_diff(sys.argv[3])
yaml_ops = read_yaml_ops()
approve_api_msg = []
approve_yaml_msg = []
api_add = []
api_delete = []
for each_diff in api_diffs:
    if each_diff[0] == '+':
        api_add.append(each_diff)
    if each_diff[0] == '-':
        api_delete.append(each_diff)
add_exclude = []
delete_exclude = []
for each_add_diff in api_add:
    for each_delete_diff in api_delete:
        if get_api_name(each_add_diff) == get_api_name(each_delete_diff) and get_api_args(each_add_diff) == get_api_args(each_delete_diff):
            add_exclude.append(each_add_diff)
            delete_exclude.append(each_delete_diff)
for exclude_item in add_exclude:
    api_add.remove(exclude_item)
for exclude_item in delete_exclude:
    api_delete.remove(exclude_item)
yaml_add = []
yaml_delete = []
for each_diff in yaml_diffs:
    if each_diff[0] == '+':
        yaml_add.append(each_diff)
    if each_diff[0] == '-':
        yaml_delete.append(each_diff)
for each_add in api_add:
    add = True
    modify = False
    need_approve = True
    yaml_name_found = False
    api_name = get_api_name(each_add)
    api_args = get_api_args(each_add)
    for each_delete_api in api_delete:
        if get_api_name(each_delete_api) == api_name:
            modify = True
            add = False
    for each_delete_yaml in yaml_delete:
        if each_delete_yaml.find(api_name) != -1:
            yaml_name_found = True
    for op in yaml_ops:
        if op['op'] == api_name:
            yaml_name_found = True
            if api_args == get_yaml_op_args(op['args']):
                need_approve = False
                break
    if modify and (not yaml_name_found):
        need_approve = False
    if add and len(yaml_add) == 0:
        need_approve = False
    if need_approve:
        approve_api_msg.append(each_add)
for each_delete in api_delete:
    api_name = get_api_name(each_delete)
    api_args = get_api_args(each_delete)
    need_approve = False
    for op in yaml_ops:
        if op['op'] == api_name and api_args == get_yaml_op_args(op['args']):
            need_approve = True
            break
    if need_approve:
        approve_api_msg.append(each_delete)
if len(api_add) == 0 and len(api_delete) == 0:
    pr_apis = read_api(sys.argv[2])
    for each_diff in yaml_delete:
        if each_diff.startswith('-- op'):
            for api in pr_apis:
                if each_diff.find(get_api_name(api)) != -1:
                    approve_yaml_msg.append(each_diff)
                    break
        if each_diff.startswith('-  args'):
            yaml_op_args_str = each_diff.strip('-  args : ')
            yaml_op_args = get_yaml_op_args(yaml_op_args_str)
            for api in pr_apis:
                if get_api_args(api) == yaml_op_args:
                    approve_yaml_msg.append(each_diff)
                    break
approve_msg = []
if len(approve_api_msg) != 0:
    approve_msg = ['The APIs you changed are as follows:']
    approve_msg.extend(approve_api_msg)
if len(approve_yaml_msg) != 0:
    approve_msg = ['The Yaml File you changed are as follows:']
    approve_msg.extend(approve_yaml_msg)
print('\r\n'.join(approve_msg))