from impacket.ldap import ldapasn1 as ldapasn1_impacket
from impacket.ldap import ldap as ldap_impacket

class CMEModule:
    """
    Created as a contributtion from HackTheBox Academy team for CrackMapExec
    Reference: https://academy.hackthebox.com/module/details/84

    Module by @juliourena
    Based on: https://github.com/juliourena/CrackMapExec/blob/master/cme/modules/get_description.py
    """
    name = 'groupmembership'
    description = 'Query the groups to which a user belongs.'
    supported_protocols = ['ldap']
    opsec_safe = True
    multiple_hosts = True

    def options(self, context, module_options):
        if False:
            i = 10
            return i + 15
        '\n        USER\tChoose a username to query group membership\n        '
        self.user = ''
        if 'USER' in module_options:
            if module_options['USER'] == '':
                context.log.fail('Invalid value for USER option!')
                exit(1)
            self.user = module_options['USER']
        else:
            context.log.fail('Missing USER option, use --options to list available parameters')
            exit(1)

    def on_login(self, context, connection):
        if False:
            i = 10
            return i + 15
        'Concurrent. Required if on_admin_login is not present. This gets called on each authenticated connection'
        searchFilter = '(&(objectClass=user)(sAMAccountName={}))'.format(self.user)
        try:
            context.log.debug('Search Filter=%s' % searchFilter)
            resp = connection.ldapConnection.search(searchFilter=searchFilter, attributes=['memberOf', 'primaryGroupID'], sizeLimit=0)
        except ldap_impacket.LDAPSearchError as e:
            if e.getErrorString().find('sizeLimitExceeded') >= 0:
                context.log.debug('sizeLimitExceeded exception caught, giving up and processing the data received')
                resp = e.getAnswers()
                pass
            else:
                context.log.debug(e)
                return False
        memberOf = []
        primaryGroupID = ''
        context.log.debug('Total of records returned %d' % len(resp))
        for item in resp:
            if isinstance(item, ldapasn1_impacket.SearchResultEntry) is not True:
                continue
            try:
                for attribute in item['attributes']:
                    if str(attribute['type']) == 'primaryGroupID':
                        primaryGroupID = attribute['vals'][0]
                        if str(primaryGroupID) == '513':
                            memberOf.append('CN=Domain Users,CN=Users,DC=XXXXX,DC=XXX')
                    elif str(attribute['type']) == 'memberOf':
                        for group in attribute['vals']:
                            if isinstance(group._value, bytes):
                                memberOf.append(str(group))
            except Exception as e:
                context.log.debug('Exception:', exc_info=True)
                context.log.debug('Skipping item, cannot process due to error %s' % str(e))
                pass
        if len(memberOf) > 0:
            context.log.success('User: {} is member of following groups: '.format(self.user))
            for group in memberOf:
                group_parts = group.split(',')
                group_name = group_parts[0].split('=')[1]
                context.log.highlight('{}'.format(group_name))