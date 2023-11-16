class CMEModule:
    """
    Module by Shutdown and Podalirius

    Initial module:
      https://github.com/ShutdownRepo/CrackMapExec-MachineAccountQuota

    Authors:
      Shutdown: @_nwodtuhs
      Podalirius: @podalirius_
    """

    def options(self, context, module_options):
        if False:
            for i in range(10):
                print('nop')
        pass
    name = 'MAQ'
    description = 'Retrieves the MachineAccountQuota domain-level attribute'
    supported_protocols = ['ldap']
    opsec_safe = True
    multiple_hosts = False

    def on_login(self, context, connection):
        if False:
            i = 10
            return i + 15
        result = []
        context.log.display('Getting the MachineAccountQuota')
        searchFilter = '(objectClass=*)'
        attributes = ['ms-DS-MachineAccountQuota']
        result = connection.search(searchFilter, attributes)
        context.log.highlight('MachineAccountQuota: %d' % result[0]['attributes'][0]['vals'][0])