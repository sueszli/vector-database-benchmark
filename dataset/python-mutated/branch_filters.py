NON_PR_BRANCH_LIST = ['main', 'master', '/ci-all\\/.*/', '/release\\/.*/']
PR_BRANCH_LIST = ['/gh\\/.*\\/head/', '/pull\\/.*/']
RC_PATTERN = '/v[0-9]+(\\.[0-9]+)*-rc[0-9]+/'
MAC_IOS_EXCLUSION_LIST = ['nightly', 'postnightly']

def gen_filter_dict(branches_list=NON_PR_BRANCH_LIST, tags_list=None):
    if False:
        return 10
    "Generates a filter dictionary for use with CircleCI's job filter"
    filter_dict = {'branches': {'only': branches_list}}
    if tags_list is not None:
        filter_dict['tags'] = {'only': tags_list}
    return filter_dict

def gen_filter_dict_exclude(branches_list=MAC_IOS_EXCLUSION_LIST):
    if False:
        print('Hello World!')
    return {'branches': {'ignore': branches_list}}