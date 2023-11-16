from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: yum_repository\nauthor: Jiri Tyr (@jtyr)\nversion_added: \'2.1\'\nshort_description: Add or remove YUM repositories\ndescription:\n  - Add or remove YUM repositories in RPM-based Linux distributions.\n  - If you wish to update an existing repository definition use M(community.general.ini_file) instead.\n\noptions:\n  async:\n    description:\n      - If set to V(true) Yum will download packages and metadata from this\n        repo in parallel, if possible.\n      - In ansible-core 2.11, 2.12, and 2.13 the default value is V(true).\n      - This option has been deprecated in RHEL 8. If you\'re using one of the\n        versions listed above, you can set this option to None to avoid passing an\n        unknown configuration option.\n    type: bool\n  bandwidth:\n    description:\n      - Maximum available network bandwidth in bytes/second. Used with the\n        O(throttle) option.\n      - If O(throttle) is a percentage and bandwidth is V(0) then bandwidth\n        throttling will be disabled. If O(throttle) is expressed as a data rate\n        (bytes/sec) then this option is ignored. Default is V(0) (no bandwidth\n        throttling).\n    type: str\n  baseurl:\n    description:\n      - URL to the directory where the yum repository\'s \'repodata\' directory\n        lives.\n      - It can also be a list of multiple URLs.\n      - This, the O(metalink) or O(mirrorlist) parameters are required if O(state) is set to\n        V(present).\n    type: list\n    elements: str\n  cost:\n    description:\n      - Relative cost of accessing this repository. Useful for weighing one\n        repo\'s packages as greater/less than any other.\n    type: str\n  deltarpm_metadata_percentage:\n    description:\n      - When the relative size of deltarpm metadata vs pkgs is larger than\n        this, deltarpm metadata is not downloaded from the repo. Note that you\n        can give values over V(100), so V(200) means that the metadata is\n        required to be half the size of the packages. Use V(0) to turn off\n        this check, and always download metadata.\n    type: str\n  deltarpm_percentage:\n    description:\n      - When the relative size of delta vs pkg is larger than this, delta is\n        not used. Use V(0) to turn off delta rpm processing. Local repositories\n        (with file://O(baseurl)) have delta rpms turned off by default.\n    type: str\n  description:\n    description:\n      - A human-readable string describing the repository. This option corresponds to the "name" property in the repo file.\n      - This parameter is only required if O(state) is set to V(present).\n    type: str\n  enabled:\n    description:\n      - This tells yum whether or not use this repository.\n      - Yum default value is V(true).\n    type: bool\n  enablegroups:\n    description:\n      - Determines whether yum will allow the use of package groups for this\n        repository.\n      - Yum default value is V(true).\n    type: bool\n  exclude:\n    description:\n      - List of packages to exclude from updates or installs. This should be a\n        space separated list. Shell globs using wildcards (for example V(*) and V(?))\n        are allowed.\n      - The list can also be a regular YAML array.\n    type: list\n    elements: str\n  failovermethod:\n    choices: [roundrobin, priority]\n    description:\n      - V(roundrobin) randomly selects a URL out of the list of URLs to start\n        with and proceeds through each of them as it encounters a failure\n        contacting the host.\n      - V(priority) starts from the first O(baseurl) listed and reads through\n        them sequentially.\n    type: str\n  file:\n    description:\n      - File name without the C(.repo) extension to save the repo in. Defaults\n        to the value of O(name).\n    type: str\n  gpgcakey:\n    description:\n      - A URL pointing to the ASCII-armored CA key file for the repository.\n    type: str\n  gpgcheck:\n    description:\n      - Tells yum whether or not it should perform a GPG signature check on\n        packages.\n      - No default setting. If the value is not set, the system setting from\n        C(/etc/yum.conf) or system default of V(false) will be used.\n    type: bool\n  gpgkey:\n    description:\n      - A URL pointing to the ASCII-armored GPG key file for the repository.\n      - It can also be a list of multiple URLs.\n    type: list\n    elements: str\n  module_hotfixes:\n    description:\n      - Disable module RPM filtering and make all RPMs from the repository\n        available. The default is V(None).\n    version_added: \'2.11\'\n    type: bool\n  http_caching:\n    description:\n      - Determines how upstream HTTP caches are instructed to handle any HTTP\n        downloads that Yum does.\n      - V(all) means that all HTTP downloads should be cached.\n      - V(packages) means that only RPM package downloads should be cached (but\n         not repository metadata downloads).\n      - V(none) means that no HTTP downloads should be cached.\n    choices: [all, packages, none]\n    type: str\n  include:\n    description:\n      - Include external configuration file. Both, local path and URL is\n        supported. Configuration file will be inserted at the position of the\n        C(include=) line. Included files may contain further include lines.\n        Yum will abort with an error if an inclusion loop is detected.\n    type: str\n  includepkgs:\n    description:\n      - List of packages you want to only use from a repository. This should be\n        a space separated list. Shell globs using wildcards (for example V(*) and V(?))\n        are allowed. Substitution variables (for example V($releasever)) are honored\n        here.\n      - The list can also be a regular YAML array.\n    type: list\n    elements: str\n  ip_resolve:\n    description:\n      - Determines how yum resolves host names.\n      - V(4) or V(IPv4) - resolve to IPv4 addresses only.\n      - V(6) or V(IPv6) - resolve to IPv6 addresses only.\n    choices: [\'4\', \'6\', IPv4, IPv6, whatever]\n    type: str\n  keepalive:\n    description:\n      - This tells yum whether or not HTTP/1.1 keepalive should be used with\n        this repository. This can improve transfer speeds by using one\n        connection when downloading multiple files from a repository.\n    type: bool\n  keepcache:\n    description:\n      - Either V(1) or V(0). Determines whether or not yum keeps the cache of\n        headers and packages after successful installation.\n      - This parameter is deprecated and will be removed in version 2.20.\n    choices: [\'0\', \'1\']\n    type: str\n  metadata_expire:\n    description:\n      - Time (in seconds) after which the metadata will expire.\n      - Default value is 6 hours.\n    type: str\n  metadata_expire_filter:\n    description:\n      - Filter the O(metadata_expire) time, allowing a trade of speed for\n        accuracy if a command doesn\'t require it. Each yum command can specify\n        that it requires a certain level of timeliness quality from the remote\n        repos. from "I\'m about to install/upgrade, so this better be current"\n        to "Anything that\'s available is good enough".\n      - V(never) - Nothing is filtered, always obey O(metadata_expire).\n      - V(read-only:past) - Commands that only care about past information are\n        filtered from metadata expiring. Eg. C(yum history) info (if history\n        needs to lookup anything about a previous transaction, then by\n        definition the remote package was available in the past).\n      - V(read-only:present) - Commands that are balanced between past and\n        future. Eg. C(yum list yum).\n      - V(read-only:future) - Commands that are likely to result in running\n        other commands which will require the latest metadata. Eg.\n        C(yum check-update).\n      - Note that this option does not override "yum clean expire-cache".\n    choices: [never, \'read-only:past\', \'read-only:present\', \'read-only:future\']\n    type: str\n  metalink:\n    description:\n      - Specifies a URL to a metalink file for the repomd.xml, a list of\n        mirrors for the entire repository are generated by converting the\n        mirrors for the repomd.xml file to a O(baseurl).\n      - This, the O(baseurl) or O(mirrorlist) parameters are required if O(state) is set to\n        V(present).\n    type: str\n  mirrorlist:\n    description:\n      - Specifies a URL to a file containing a list of baseurls.\n      - This, the O(baseurl) or O(metalink) parameters are required if O(state) is set to\n        V(present).\n    type: str\n  mirrorlist_expire:\n    description:\n      - Time (in seconds) after which the mirrorlist locally cached will\n        expire.\n      - Default value is 6 hours.\n    type: str\n  name:\n    description:\n      - Unique repository ID. This option builds the section name of the repository in the repo file.\n      - This parameter is only required if O(state) is set to V(present) or\n        V(absent).\n    type: str\n    required: true\n  password:\n    description:\n      - Password to use with the username for basic authentication.\n    type: str\n  priority:\n    description:\n      - Enforce ordered protection of repositories. The value is an integer\n        from 1 to 99.\n      - This option only works if the YUM Priorities plugin is installed.\n    type: str\n  protect:\n    description:\n      - Protect packages from updates from other repositories.\n    type: bool\n  proxy:\n    description:\n      - URL to the proxy server that yum should use. Set to V(_none_) to\n        disable the global proxy setting.\n    type: str\n  proxy_password:\n    description:\n      - Password for this proxy.\n    type: str\n  proxy_username:\n    description:\n      - Username to use for proxy.\n    type: str\n  repo_gpgcheck:\n    description:\n      - This tells yum whether or not it should perform a GPG signature check\n        on the repodata from this repository.\n    type: bool\n  reposdir:\n    description:\n      - Directory where the C(.repo) files will be stored.\n    type: path\n    default: /etc/yum.repos.d\n  retries:\n    description:\n      - Set the number of times any attempt to retrieve a file should retry\n        before returning an error. Setting this to V(0) makes yum try forever.\n    type: str\n  s3_enabled:\n    description:\n      - Enables support for S3 repositories.\n      - This option only works if the YUM S3 plugin is installed.\n    type: bool\n  skip_if_unavailable:\n    description:\n      - If set to V(true) yum will continue running if this repository cannot be\n        contacted for any reason. This should be set carefully as all repos are\n        consulted for any given command.\n    type: bool\n  ssl_check_cert_permissions:\n    description:\n      - Whether yum should check the permissions on the paths for the\n        certificates on the repository (both remote and local).\n      - If we can\'t read any of the files then yum will force\n        O(skip_if_unavailable) to be V(true). This is most useful for non-root\n        processes which use yum on repos that have client cert files which are\n        readable only by root.\n    type: bool\n  sslcacert:\n    description:\n      - Path to the directory containing the databases of the certificate\n        authorities yum should use to verify SSL certificates.\n    type: str\n    aliases: [ ca_cert ]\n  sslclientcert:\n    description:\n      - Path to the SSL client certificate yum should use to connect to\n        repos/remote sites.\n    type: str\n    aliases: [ client_cert ]\n  sslclientkey:\n    description:\n      - Path to the SSL client key yum should use to connect to repos/remote\n        sites.\n    type: str\n    aliases: [ client_key ]\n  sslverify:\n    description:\n      - Defines whether yum should verify SSL certificates/hosts at all.\n    type: bool\n    aliases: [ validate_certs ]\n  state:\n    description:\n      - State of the repo file.\n    choices: [absent, present]\n    type: str\n    default: present\n  throttle:\n    description:\n      - Enable bandwidth throttling for downloads.\n      - This option can be expressed as a absolute data rate in bytes/sec. An\n        SI prefix (k, M or G) may be appended to the bandwidth value.\n    type: str\n  timeout:\n    description:\n      - Number of seconds to wait for a connection before timing out.\n    type: str\n  ui_repoid_vars:\n    description:\n      - When a repository id is displayed, append these yum variables to the\n        string if they are used in the O(baseurl)/etc. Variables are appended\n        in the order listed (and found).\n    type: str\n  username:\n    description:\n      - Username to use for basic authentication to a repo or really any url.\n    type: str\n\nextends_documentation_fragment:\n    - action_common_attributes\n    - files\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: full\n    platform:\n        platforms: rhel\nnotes:\n  - All comments will be removed if modifying an existing repo file.\n  - Section order is preserved in an existing repo file.\n  - Parameters in a section are ordered alphabetically in an existing repo\n    file.\n  - The repo file will be automatically deleted if it contains no repository.\n  - When removing a repository, beware that the metadata cache may still remain\n    on disk until you run C(yum clean all). Use a notification handler for this.\n  - "The O(ignore:params) parameter was removed in Ansible 2.5 due to circumventing Ansible\'s parameter\n    handling"\n'
EXAMPLES = '\n- name: Add repository\n  ansible.builtin.yum_repository:\n    name: epel\n    description: EPEL YUM repo\n    baseurl: https://download.fedoraproject.org/pub/epel/$releasever/$basearch/\n\n- name: Add multiple repositories into the same file (1/2)\n  ansible.builtin.yum_repository:\n    name: epel\n    description: EPEL YUM repo\n    file: external_repos\n    baseurl: https://download.fedoraproject.org/pub/epel/$releasever/$basearch/\n    gpgcheck: no\n\n- name: Add multiple repositories into the same file (2/2)\n  ansible.builtin.yum_repository:\n    name: rpmforge\n    description: RPMforge YUM repo\n    file: external_repos\n    baseurl: http://apt.sw.be/redhat/el7/en/$basearch/rpmforge\n    mirrorlist: http://mirrorlist.repoforge.org/el7/mirrors-rpmforge\n    enabled: no\n\n# Handler showing how to clean yum metadata cache\n- name: yum-clean-metadata\n  ansible.builtin.command: yum clean metadata\n\n# Example removing a repository and cleaning up metadata cache\n- name: Remove repository (and clean up left-over metadata)\n  ansible.builtin.yum_repository:\n    name: epel\n    state: absent\n  notify: yum-clean-metadata\n\n- name: Remove repository from a specific repo file\n  ansible.builtin.yum_repository:\n    name: epel\n    file: external_repos\n    state: absent\n'
RETURN = '\nrepo:\n    description: repository name\n    returned: success\n    type: str\n    sample: "epel"\nstate:\n    description: state of the target, after execution\n    returned: success\n    type: str\n    sample: "present"\n'
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.common.text.converters import to_native

class YumRepo(object):
    module = None
    params = None
    section = None
    repofile = configparser.RawConfigParser()
    allowed_params = ['async', 'bandwidth', 'baseurl', 'cost', 'deltarpm_metadata_percentage', 'deltarpm_percentage', 'enabled', 'enablegroups', 'exclude', 'failovermethod', 'gpgcakey', 'gpgcheck', 'gpgkey', 'module_hotfixes', 'http_caching', 'include', 'includepkgs', 'ip_resolve', 'keepalive', 'keepcache', 'metadata_expire', 'metadata_expire_filter', 'metalink', 'mirrorlist', 'mirrorlist_expire', 'name', 'password', 'priority', 'protect', 'proxy', 'proxy_password', 'proxy_username', 'repo_gpgcheck', 'retries', 's3_enabled', 'skip_if_unavailable', 'sslcacert', 'ssl_check_cert_permissions', 'sslclientcert', 'sslclientkey', 'sslverify', 'throttle', 'timeout', 'ui_repoid_vars', 'username']
    list_params = ['exclude', 'includepkgs']

    def __init__(self, module):
        if False:
            print('Hello World!')
        self.module = module
        self.params = self.module.params
        self.section = self.params['repoid']
        repos_dir = self.params['reposdir']
        if not os.path.isdir(repos_dir):
            self.module.fail_json(msg="Repo directory '%s' does not exist." % repos_dir)
        self.params['dest'] = os.path.join(repos_dir, '%s.repo' % self.params['file'])
        if os.path.isfile(self.params['dest']):
            self.repofile.read(self.params['dest'])

    def add(self):
        if False:
            i = 10
            return i + 15
        if self.repofile.has_section(self.section):
            self.repofile.remove_section(self.section)
        self.repofile.add_section(self.section)
        req_params = (self.params['baseurl'], self.params['metalink'], self.params['mirrorlist'])
        if req_params == (None, None, None):
            self.module.fail_json(msg="Parameter 'baseurl', 'metalink' or 'mirrorlist' is required for adding a new repo.")
        for (key, value) in sorted(self.params.items()):
            if key in self.list_params and isinstance(value, list):
                value = ' '.join(value)
            elif isinstance(value, bool):
                value = int(value)
            if value is not None and key in self.allowed_params:
                if key == 'keepcache':
                    self.module.deprecate("'keepcache' parameter is deprecated.", version='2.20')
                self.repofile.set(self.section, key, value)

    def save(self):
        if False:
            while True:
                i = 10
        if len(self.repofile.sections()):
            try:
                with open(self.params['dest'], 'w') as fd:
                    self.repofile.write(fd)
            except IOError as e:
                self.module.fail_json(msg='Problems handling file %s.' % self.params['dest'], details=to_native(e))
        else:
            try:
                os.remove(self.params['dest'])
            except OSError as e:
                self.module.fail_json(msg='Cannot remove empty repo file %s.' % self.params['dest'], details=to_native(e))

    def remove(self):
        if False:
            i = 10
            return i + 15
        if self.repofile.has_section(self.section):
            self.repofile.remove_section(self.section)

    def dump(self):
        if False:
            i = 10
            return i + 15
        repo_string = ''
        for section in sorted(self.repofile.sections()):
            repo_string += '[%s]\n' % section
            for (key, value) in sorted(self.repofile.items(section)):
                repo_string += '%s = %s\n' % (key, value)
            repo_string += '\n'
        return repo_string

def main():
    if False:
        while True:
            i = 10
    argument_spec = dict(bandwidth=dict(), baseurl=dict(type='list', elements='str'), cost=dict(), deltarpm_metadata_percentage=dict(), deltarpm_percentage=dict(), description=dict(), enabled=dict(type='bool'), enablegroups=dict(type='bool'), exclude=dict(type='list', elements='str'), failovermethod=dict(choices=['roundrobin', 'priority']), file=dict(), gpgcakey=dict(no_log=False), gpgcheck=dict(type='bool'), gpgkey=dict(type='list', elements='str', no_log=False), module_hotfixes=dict(type='bool'), http_caching=dict(choices=['all', 'packages', 'none']), include=dict(), includepkgs=dict(type='list', elements='str'), ip_resolve=dict(choices=['4', '6', 'IPv4', 'IPv6', 'whatever']), keepalive=dict(type='bool'), keepcache=dict(choices=['0', '1']), metadata_expire=dict(), metadata_expire_filter=dict(choices=['never', 'read-only:past', 'read-only:present', 'read-only:future']), metalink=dict(), mirrorlist=dict(), mirrorlist_expire=dict(), name=dict(required=True), password=dict(no_log=True), priority=dict(), protect=dict(type='bool'), proxy=dict(), proxy_password=dict(no_log=True), proxy_username=dict(), repo_gpgcheck=dict(type='bool'), reposdir=dict(default='/etc/yum.repos.d', type='path'), retries=dict(), s3_enabled=dict(type='bool'), skip_if_unavailable=dict(type='bool'), sslcacert=dict(aliases=['ca_cert']), ssl_check_cert_permissions=dict(type='bool'), sslclientcert=dict(aliases=['client_cert']), sslclientkey=dict(aliases=['client_key'], no_log=False), sslverify=dict(type='bool', aliases=['validate_certs']), state=dict(choices=['present', 'absent'], default='present'), throttle=dict(), timeout=dict(), ui_repoid_vars=dict(), username=dict())
    argument_spec['async'] = dict(type='bool')
    module = AnsibleModule(argument_spec=argument_spec, add_file_common_args=True, supports_check_mode=True)
    name = module.params['name']
    state = module.params['state']
    if state == 'present':
        if module.params['baseurl'] is None and module.params['metalink'] is None and (module.params['mirrorlist'] is None):
            module.fail_json(msg="Parameter 'baseurl', 'metalink' or 'mirrorlist' is required.")
        if module.params['description'] is None:
            module.fail_json(msg="Parameter 'description' is required.")
    module.params['repoid'] = module.params['name']
    module.params['name'] = module.params['description']
    del module.params['description']
    for list_param in ['baseurl', 'gpgkey']:
        if list_param in module.params and module.params[list_param] is not None:
            module.params[list_param] = '\n'.join(module.params[list_param])
    if module.params['file'] is None:
        module.params['file'] = module.params['repoid']
    yumrepo = YumRepo(module)
    diff = {'before_header': yumrepo.params['dest'], 'before': yumrepo.dump(), 'after_header': yumrepo.params['dest'], 'after': ''}
    if state == 'present':
        yumrepo.add()
    elif state == 'absent':
        yumrepo.remove()
    diff['after'] = yumrepo.dump()
    changed = diff['before'] != diff['after']
    if not module.check_mode and changed:
        yumrepo.save()
    if os.path.isfile(module.params['dest']):
        file_args = module.load_file_common_arguments(module.params)
        changed = module.set_fs_attributes_if_different(file_args, changed)
    module.exit_json(changed=changed, repo=name, state=state, diff=diff)
if __name__ == '__main__':
    main()