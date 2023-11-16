from pathlib import Path
from datetime import datetime
from impacket.ldap import ldap, ldapasn1
from impacket.ldap.ldap import LDAPSearchError

class CMEModule:
    """
    Get user descriptions stored in Active Directory.

    Module by Tobias Neitzel (@qtc_de)
    """
    name = 'user-desc'
    description = 'Get user descriptions stored in Active Directory'
    supported_protocols = ['ldap']
    opsec_safe = True
    multiple_hosts = True

    def __init__(self, context=None, multiple_options=None):
        if False:
            return 10
        self.keywords = None
        self.search_filter = None
        self.account_names = None
        self.context = None
        self.desc_count = None
        self.log_file = None

    def options(self, context, module_options):
        if False:
            i = 10
            return i + 15
        '\n        LDAP_FILTER     Custom LDAP search filter (fully replaces the default search)\n        DESC_FILTER     An additional seach filter for descriptions (supports wildcard *)\n        DESC_INVERT     An additional seach filter for descriptions (shows non matching)\n        USER_FILTER     An additional seach filter for usernames (supports wildcard *)\n        USER_INVERT     An additional seach filter for usernames (shows non matching)\n        KEYWORDS        Use a custom set of keywords (comma separated)\n        ADD_KEYWORDS    Add additional keywords to the default set (comma separated)\n        '
        self.log_file = None
        self.desc_count = 0
        self.context = context
        self.account_names = set()
        self.keywords = {'pass', 'creds', 'creden', 'key', 'secret', 'default'}
        if 'LDAP_FILTER' in module_options:
            self.search_filter = module_options['LDAP_FILTER']
        else:
            self.search_filter = '(&(objectclass=user)'
            if 'DESC_FILTER' in module_options:
                self.search_filter += f"(description={module_options['DESC_FILTER']})"
            if 'DESC_INVERT' in module_options:
                self.search_filter += f"(!(description={module_options['DESC_INVERT']}))"
            if 'USER_FILTER' in module_options:
                self.search_filter += f"(sAMAccountName={module_options['USER_FILTER']})"
            if 'USER_INVERT' in module_options:
                self.search_filter += f"(!(sAMAccountName={module_options['USER_INVERT']}))"
            self.search_filter += ')'
        if 'KEYWORDS' in module_options:
            self.keywords = set(module_options['KEYWORDS'].split(','))
        elif 'ADD_KEYWORDS' in module_options:
            add_keywords = set(module_options['ADD_KEYWORDS'].split(','))
            self.keywords = self.keywords.union(add_keywords)

    def on_login(self, context, connection):
        if False:
            return 10
        '\n        On successful LDAP login we perform a search for all user objects that have a description.\n        Users can specify additional LDAP filters that are applied to the query.\n        '
        self.create_log_file(connection.conn.getRemoteHost(), datetime.now().strftime('%Y%m%d_%H%M%S'))
        context.log.info(f"Starting LDAP search with search filter '{self.search_filter}'")
        try:
            sc = ldap.SimplePagedResultsControl()
            connection.ldapConnection.search(searchFilter=self.search_filter, attributes=['sAMAccountName', 'description'], sizeLimit=0, searchControls=[sc], perRecordCallback=self.process_record)
        except LDAPSearchError as e:
            context.log.fail(f'Obtained unexpected exception: {str(e)}')
        finally:
            self.delete_log_file()

    def create_log_file(self, host, time):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a log file for dumping user descriptions.\n        '
        logfile = f'UserDesc-{host}-{time}.log'
        logfile = Path.home().joinpath('.cme').joinpath('logs').joinpath(logfile)
        self.context.log.info(f"Creating log file '{logfile}'")
        self.log_file = open(logfile, 'w')
        self.append_to_log('User:', 'Description:')

    def delete_log_file(self):
        if False:
            return 10
        '\n        Closes the log file.\n        '
        try:
            self.log_file.close()
            info = f'Saved {self.desc_count} user descriptions to {self.log_file.name}'
            self.context.log.highlight(info)
        except AttributeError:
            pass

    def append_to_log(self, user, description):
        if False:
            return 10
        '\n        Append a new entry to the log file. Helper function that is only used to have an\n        unified padding on the user field.\n        '
        print(user.ljust(25), description, file=self.log_file)

    def process_record(self, item):
        if False:
            return 10
        '\n        Function that is called to process the items obtained by the LDAP search. All items are\n        written to the log file per default. Items that contain one of the keywords configured\n        within this module are also printed to stdout.\n\n        On large Active Directories there seems to be a problem with duplicate user entries. For\n        some reason the process_record function is called multiple times with the same user entry.\n        Not sure whether this is a fault by this module or by impacket. As a workaround, this\n        function adds each new account name to a set and skips accounts that have already been added.\n        '
        if not isinstance(item, ldapasn1.SearchResultEntry):
            return
        sAMAccountName = ''
        description = ''
        try:
            for attribute in item['attributes']:
                if str(attribute['type']) == 'sAMAccountName':
                    sAMAccountName = attribute['vals'][0].asOctets().decode('utf-8')
                elif str(attribute['type']) == 'description':
                    description = attribute['vals'][0].asOctets().decode('utf-8')
        except Exception as e:
            entry = sAMAccountName or 'item'
            self.context.error(f"Skipping {entry}, cannot process LDAP entry due to error: '{str(e)}'")
        if description and sAMAccountName not in self.account_names:
            self.desc_count += 1
            self.append_to_log(sAMAccountName, description)
            if self.highlight(description):
                self.context.log.highlight(f'User: {sAMAccountName} - Description: {description}')
            self.account_names.add(sAMAccountName)

    def highlight(self, description):
        if False:
            i = 10
            return i + 15
        '\n        Check for interesting entries. Just checks whether certain keywords are contained within the\n        user description. Keywords are configured at the top of this class within the options function.\n\n        It is tempting to implement more logic here (e.g. catch all strings that are longer than seven\n        characters and contain 3 different character classes). Such functionality is nice when playing\n        CTF in small AD environments. When facing a real AD, such functionality gets annoying, because\n        it generates too much output with 99% of it being false positives.\n\n        The recommended way when targeting user descriptions is to use the keyword filter to catch low-hanging fruit.\n        More dedicated searches for sensitive information should be done using the logfile.\n        This allows you to refine your search query at any time without having to pull data from AD again.\n        '
        for keyword in self.keywords:
            if keyword.lower() in description.lower():
                return True
        return False