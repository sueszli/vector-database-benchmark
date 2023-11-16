from __future__ import division
from __future__ import print_function
import pytest
import unittest
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5 import lsad
from impacket.dcerpc.v5.ndr import NULL
from impacket.dcerpc.v5.dtypes import MAXIMUM_ALLOWED, RPC_UNICODE_STRING, DELETE
from impacket.structure import hexdump

class LSADTests(DCERPCTests):
    iface_uuid = lsad.MSRPC_UUID_LSAD
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\lsarpc]'
    authn = True

    def open_policy(self, dce):
        if False:
            print('Hello World!')
        resp = lsad.hLsarOpenPolicy2(dce, MAXIMUM_ALLOWED | lsad.POLICY_CREATE_SECRET | DELETE | lsad.POLICY_VIEW_LOCAL_INFORMATION)
        return resp['PolicyHandle']

    def test_LsarOpenPolicy(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = lsad.LsarOpenPolicy()
        request['SystemName'] = NULL
        request['ObjectAttributes']['RootDirectory'] = NULL
        request['ObjectAttributes']['ObjectName'] = NULL
        request['ObjectAttributes']['SecurityDescriptor'] = NULL
        request['ObjectAttributes']['SecurityQualityOfService'] = NULL
        request['DesiredAccess'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_hLsarOpenPolicy(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        resp = lsad.hLsarOpenPolicy(dce)
        resp.dump()

    def test_LsarQueryInformationPolicy2(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQueryInformationPolicy2()
        request['PolicyHandle'] = policyHandle
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAuditLogInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyPrimaryDomainInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyPdAccountInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyLsaServerRoleInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyReplicaSourceInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyDnsDomainInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyDnsDomainInformationInt
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyLocalAccountDomainInformation
        resp = dce.request(request)
        resp.dump()

    def test_hLsarQueryInformationPolicy2(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditLogInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyPrimaryDomainInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyPdAccountInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyLsaServerRoleInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyReplicaSourceInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyDnsDomainInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyDnsDomainInformationInt)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyLocalAccountDomainInformation)
        resp.dump()

    def test_LsarQueryInformationPolicy(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQueryInformationPolicy()
        request['PolicyHandle'] = policyHandle
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAuditLogInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyPrimaryDomainInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyPdAccountInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyLsaServerRoleInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyReplicaSourceInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyDnsDomainInformation
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyDnsDomainInformationInt
        resp = dce.request(request)
        resp.dump()
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyLocalAccountDomainInformation
        resp = dce.request(request)
        resp.dump()

    def test_hLsarQueryInformationPolicy(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditLogInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyPrimaryDomainInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyPdAccountInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyLsaServerRoleInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyReplicaSourceInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyDnsDomainInformation)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyDnsDomainInformationInt)
        resp.dump()
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyLocalAccountDomainInformation)
        resp.dump()

    def test_LsarQueryDomainInformationPolicy(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQueryDomainInformationPolicy()
        request['PolicyHandle'] = policyHandle
        request['InformationClass'] = lsad.POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainQualityOfServiceInformation
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_INVALID_PARAMETER') < 0:
                raise
        request['InformationClass'] = lsad.POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainEfsInformation
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_OBJECT_NAME_NOT_FOUND') < 0:
                raise
        request['InformationClass'] = lsad.POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainKerberosTicketInformation
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_OBJECT_NAME_NOT_FOUND') < 0:
                raise

    def test_hLsarQueryDomainInformationPolicy(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        try:
            resp = lsad.hLsarQueryDomainInformationPolicy(dce, policyHandle, lsad.POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainQualityOfServiceInformation)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_INVALID_PARAMETER') < 0:
                raise
        try:
            resp = lsad.hLsarQueryDomainInformationPolicy(dce, policyHandle, lsad.POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainEfsInformation)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_OBJECT_NAME_NOT_FOUND') < 0:
                raise
        try:
            resp = lsad.hLsarQueryDomainInformationPolicy(dce, policyHandle, lsad.POLICY_DOMAIN_INFORMATION_CLASS.PolicyDomainKerberosTicketInformation)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_OBJECT_NAME_NOT_FOUND') < 0:
                raise

    def test_LsarEnumerateAccounts(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarEnumerateAccounts()
        request['PolicyHandle'] = policyHandle
        request['PreferedMaximumLength'] = 4294967295
        resp = dce.request(request)
        resp.dump()

    def test_hLsarEnumerateAccounts(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarEnumerateAccounts(dce, policyHandle)
        resp.dump()

    def test_LsarEnumerateAccountsWithUserRight(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarEnumerateAccountsWithUserRight()
        request['PolicyHandle'] = policyHandle
        request['UserRight'] = 'SeSystemtimePrivilege'
        resp = dce.request(request)
        resp.dump()

    def test_hLsarEnumerateAccountsWithUserRight(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarEnumerateAccountsWithUserRight(dce, policyHandle, 'SeSystemtimePrivilege')
        resp.dump()

    def test_LsarEnumerateTrustedDomainsEx(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarEnumerateTrustedDomainsEx()
        request['PolicyHandle'] = policyHandle
        request['EnumerationContext'] = 0
        request['PreferedMaximumLength'] = 4294967295
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_NO_MORE_ENTRIES') < 0:
                raise

    def test_hLsarEnumerateTrustedDomainsEx(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        try:
            resp = lsad.hLsarEnumerateTrustedDomainsEx(dce, policyHandle)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_NO_MORE_ENTRIES') < 0:
                raise

    def test_LsarEnumerateTrustedDomains(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarEnumerateTrustedDomains()
        request['PolicyHandle'] = policyHandle
        request['EnumerationContext'] = 0
        request['PreferedMaximumLength'] = 4294967295
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_NO_MORE_ENTRIES') < 0:
                raise

    def test_hLsarEnumerateTrustedDomains(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        try:
            resp = lsad.hLsarEnumerateTrustedDomains(dce, policyHandle)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_NO_MORE_ENTRIES') < 0:
                raise

    def test_hLsarOpenAccount(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarEnumerateAccounts(dce, policyHandle)
        resp.dump()
        resp = lsad.hLsarOpenAccount(dce, policyHandle, resp['EnumerationBuffer']['Information'][0]['Sid'].formatCanonical())
        resp.dump()
        resp = lsad.hLsarClose(dce, resp['AccountHandle'])
        resp.dump()

    def test_LsarOpenAccount(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarEnumerateAccounts()
        request['PolicyHandle'] = policyHandle
        request['PreferedMaximumLength'] = 4294967295
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarOpenAccount()
        request['PolicyHandle'] = policyHandle
        request['AccountSid'] = resp['EnumerationBuffer']['Information'][0]['Sid']
        request['DesiredAccess'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarClose()
        request['ObjectHandle'] = resp['AccountHandle']
        resp = dce.request(request)
        resp.dump()

    def test_LsarCreateAccount_LsarDeleteObject(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQueryInformationPolicy2()
        request['PolicyHandle'] = policyHandle
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation
        resp = dce.request(request)
        sid = resp['PolicyInformation']['PolicyAccountDomainInfo']['DomainSid'].formatCanonical()
        sid = sid + '-9999'
        request = lsad.LsarCreateAccount()
        request['PolicyHandle'] = policyHandle
        request['AccountSid'].fromCanonical(sid)
        request['DesiredAccess'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarDeleteObject()
        request['ObjectHandle'] = resp['AccountHandle']
        resp = dce.request(request)
        resp.dump()

    def test_hLsarCreateAccount_hLsarDeleteObject(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation)
        sid = resp['PolicyInformation']['PolicyAccountDomainInfo']['DomainSid'].formatCanonical()
        sid = sid + '-9999'
        resp = lsad.hLsarCreateAccount(dce, policyHandle, sid)
        resp.dump()
        resp = lsad.hLsarDeleteObject(dce, resp['AccountHandle'])
        resp.dump()

    def test_LsarEnumeratePrivilegesAccount(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        sid = 'S-1-5-32-544'
        request = lsad.LsarOpenAccount()
        request['PolicyHandle'] = policyHandle
        request['AccountSid'].fromCanonical(sid)
        request['DesiredAccess'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarEnumeratePrivilegesAccount()
        request['AccountHandle'] = resp['AccountHandle']
        resp = dce.request(request)
        resp.dump()

    def test_hLsarEnumeratePrivilegesAccount(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        sid = 'S-1-5-32-544'
        resp = lsad.hLsarOpenAccount(dce, policyHandle, sid)
        resp.dump()
        resp = lsad.hLsarEnumeratePrivilegesAccount(dce, resp['AccountHandle'])
        resp.dump()

    def test_LsarGetSystemAccessAccount_LsarSetSystemAccessAccount(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        sid = 'S-1-5-32-544'
        request = lsad.LsarOpenAccount()
        request['PolicyHandle'] = policyHandle
        request['AccountSid'].fromCanonical(sid)
        request['DesiredAccess'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarGetSystemAccessAccount()
        request['AccountHandle'] = resp['AccountHandle']
        resp2 = dce.request(request)
        resp.dump()
        request = lsad.LsarSetSystemAccessAccount()
        request['AccountHandle'] = resp['AccountHandle']
        request['SystemAccess'] = resp2['SystemAccess']
        resp = dce.request(request)
        resp.dump()

    def test_hLsarGetSystemAccessAccount_hLsarSetSystemAccessAccount(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        sid = 'S-1-5-32-544'
        resp = lsad.hLsarOpenAccount(dce, policyHandle, sid)
        resp.dump()
        resp2 = lsad.hLsarGetSystemAccessAccount(dce, resp['AccountHandle'])
        resp2.dump()
        resp = lsad.hLsarSetSystemAccessAccount(dce, resp['AccountHandle'], resp2['SystemAccess'])
        resp.dump()

    def test_LsarAddPrivilegesToAccount_LsarRemovePrivilegesFromAccount(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQueryInformationPolicy2()
        request['PolicyHandle'] = policyHandle
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation
        resp = dce.request(request)
        sid = resp['PolicyInformation']['PolicyAccountDomainInfo']['DomainSid'].formatCanonical()
        sid = sid + '-9999'
        request = lsad.LsarCreateAccount()
        request['PolicyHandle'] = policyHandle
        request['AccountSid'].fromCanonical(sid)
        request['DesiredAccess'] = MAXIMUM_ALLOWED | lsad.ACCOUNT_ADJUST_PRIVILEGES
        resp = dce.request(request)
        resp.dump()
        accountHandle = resp['AccountHandle']
        request = lsad.LsarAddPrivilegesToAccount()
        request['AccountHandle'] = accountHandle
        request['Privileges']['PrivilegeCount'] = 1
        request['Privileges']['Control'] = 0
        attribute = lsad.LSAPR_LUID_AND_ATTRIBUTES()
        attribute['Luid']['LowPart'] = 0
        attribute['Luid']['HighPart'] = 3
        attribute['Attributes'] = 3
        request['Privileges']['Privilege'].append(attribute)
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception:
            request = lsad.LsarDeleteObject()
            request['ObjectHandle'] = accountHandle
            dce.request(request)
            return
        request = lsad.LsarRemovePrivilegesFromAccount()
        request['AccountHandle'] = accountHandle
        request['AllPrivileges'] = 1
        request['Privileges'] = NULL
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarDeleteObject()
        request['ObjectHandle'] = accountHandle
        resp = dce.request(request)
        resp.dump()

    def test_hLsarAddPrivilegesToAccount_hLsarRemovePrivilegesFromAccount(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAccountDomainInformation)
        sid = resp['PolicyInformation']['PolicyAccountDomainInfo']['DomainSid'].formatCanonical()
        sid = sid + '-9999'
        resp = lsad.hLsarCreateAccount(dce, policyHandle, sid)
        accountHandle = resp['AccountHandle']
        attributes = list()
        attribute = lsad.LSAPR_LUID_AND_ATTRIBUTES()
        attribute['Luid']['LowPart'] = 0
        attribute['Luid']['HighPart'] = 3
        attribute['Attributes'] = 3
        attributes.append(attribute)
        try:
            resp = lsad.hLsarAddPrivilegesToAccount(dce, accountHandle, attributes)
            resp.dump()
        except Exception:
            resp = lsad.hLsarDeleteObject(dce, accountHandle)
            return
        resp = lsad.hLsarRemovePrivilegesFromAccount(dce, accountHandle, NULL, 1)
        resp.dump()
        resp = lsad.hLsarDeleteObject(dce, accountHandle)
        resp.dump()

    def test_LsarEnumerateAccountRights(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        sid = 'S-1-5-32-544'
        request = lsad.LsarEnumerateAccountRights()
        request['PolicyHandle'] = policyHandle
        request['AccountSid'].fromCanonical(sid)
        resp = dce.request(request)
        resp.dump()

    def test_hLsarEnumerateAccountRights(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        sid = 'S-1-5-32-544'
        resp = lsad.hLsarEnumerateAccountRights(dce, policyHandle, sid)
        resp.dump()

    def test_LsarAddAccountRights_LsarRemoveAccountRights(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        sid = 'S-1-5-32-504'
        request = lsad.LsarAddAccountRights()
        request['PolicyHandle'] = policyHandle
        request['AccountSid'].fromCanonical(sid)
        request['UserRights']['EntriesRead'] = 1
        right = RPC_UNICODE_STRING()
        right['Data'] = 'SeChangeNotifyPrivilege'
        request['UserRights']['UserRights'].append(right)
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarRemoveAccountRights()
        request['PolicyHandle'] = policyHandle
        request['AccountSid'].fromCanonical(sid)
        request['UserRights']['EntriesRead'] = 1
        right = RPC_UNICODE_STRING()
        right['Data'] = 'SeChangeNotifyPrivilege'
        request['UserRights']['UserRights'].append(right)
        resp = dce.request(request)
        resp.dump()

    def test_hLsarAddAccountRights_hLsarRemoveAccountRights(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        sid = 'S-1-5-32-504'
        resp = lsad.hLsarAddAccountRights(dce, policyHandle, sid, ('SeChangeNotifyPrivilege',))
        resp.dump()
        resp = lsad.hLsarRemoveAccountRights(dce, policyHandle, sid, ('SeChangeNotifyPrivilege',))
        resp.dump()

    def test_LsarCreateSecret_LsarOpenSecret(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarCreateSecret()
        request['PolicyHandle'] = policyHandle
        request['SecretName'] = 'MYSECRET'
        request['DesiredAccess'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarOpenSecret()
        request['PolicyHandle'] = policyHandle
        request['SecretName'] = 'MYSECRET'
        request['DesiredAccess'] = MAXIMUM_ALLOWED
        resp0 = dce.request(request)
        resp0.dump()
        request = lsad.LsarSetSecret()
        request['SecretHandle'] = resp0['SecretHandle']
        request['EncryptedCurrentValue']['Length'] = 16
        request['EncryptedCurrentValue']['MaximumLength'] = 16
        request['EncryptedCurrentValue']['Buffer'] = list('A' * 16)
        request['EncryptedOldValue']['Length'] = 16
        request['EncryptedOldValue']['MaximumLength'] = 16
        request['EncryptedOldValue']['Buffer'] = list('A' * 16)
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception:
            pass
        request = lsad.LsarDeleteObject()
        request['ObjectHandle'] = resp0['SecretHandle']
        resp = dce.request(request)
        resp.dump()

    def test_hLsarCreateSecret_hLsarOpenSecret(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarCreateSecret(dce, policyHandle, 'MYSECRET')
        resp.dump()
        resp0 = lsad.hLsarOpenSecret(dce, policyHandle, 'MYSECRET')
        resp0.dump()
        try:
            resp = lsad.hLsarSetSecret(dce, resp0['SecretHandle'], 'A' * 16, 'A' * 16)
            resp.dump()
        except Exception:
            pass
        resp = lsad.hLsarDeleteObject(dce, resp0['SecretHandle'])
        resp.dump()

    def test_LsarQuerySecret(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarOpenSecret()
        request['PolicyHandle'] = policyHandle
        request['SecretName'] = 'DPAPI_SYSTEM'
        request['DesiredAccess'] = MAXIMUM_ALLOWED
        resp0 = dce.request(request)
        resp0.dump()
        request = lsad.LsarQuerySecret()
        request['SecretHandle'] = resp0['SecretHandle']
        request['EncryptedCurrentValue']['Buffer'] = NULL
        request['EncryptedOldValue']['Buffer'] = NULL
        request['OldValueSetTime'] = NULL
        resp = dce.request(request)
        resp.dump()

    def test_hLsarQuerySecret(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp0 = lsad.hLsarOpenSecret(dce, policyHandle, 'DPAPI_SYSTEM')
        resp0.dump()
        resp = lsad.hLsarQuerySecret(dce, resp0['SecretHandle'])
        resp.dump()

    def test_LsarRetrievePrivateData_LsarStorePrivateData(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarRetrievePrivateData()
        request['PolicyHandle'] = policyHandle
        request['KeyName'] = 'DPAPI_SYSTEM'
        resp0 = dce.request(request)
        resp0.dump()
        request = lsad.LsarStorePrivateData()
        request['PolicyHandle'] = policyHandle
        request['KeyName'] = 'BETUS'
        request['EncryptedData'] = resp0['EncryptedData']
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarStorePrivateData()
        request['PolicyHandle'] = policyHandle
        request['KeyName'] = 'BETUS'
        request['EncryptedData'] = NULL
        resp = dce.request(request)
        resp.dump()

    def test_hLsarRetrievePrivateData_hLsarStorePrivateData(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp0 = lsad.hLsarRetrievePrivateData(dce, policyHandle, 'DPAPI_SYSTEM')
        resp = lsad.hLsarStorePrivateData(dce, policyHandle, 'BETUS', resp0)
        resp.dump()
        resp = lsad.hLsarStorePrivateData(dce, policyHandle, 'BETUS', NULL)
        resp.dump()

    def test_LsarEnumeratePrivileges(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarEnumeratePrivileges()
        request['PolicyHandle'] = policyHandle
        request['EnumerationContext'] = 0
        request['PreferedMaximumLength'] = 4294967295
        resp = dce.request(request)
        resp.dump()
        self.assertEqual(resp['EnumerationBuffer']['Entries'], len(resp['EnumerationBuffer']['Privileges']))

    def test_hLsarEnumeratePrivileges(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarEnumeratePrivileges(dce, policyHandle)
        resp.dump()
        self.assertEqual(resp['EnumerationBuffer']['Entries'], len(resp['EnumerationBuffer']['Privileges']))

    def test_LsarLookupPrivilegeValue_LsarLookupPrivilegeName(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarLookupPrivilegeValue()
        request['PolicyHandle'] = policyHandle
        request['Name'] = 'SeTimeZonePrivilege'
        resp = dce.request(request)
        resp.dump()
        request = lsad.LsarLookupPrivilegeName()
        request['PolicyHandle'] = policyHandle
        request['Value'] = resp['Value']
        resp = dce.request(request)
        resp.dump()
        self.assertEqual(resp['Name'], 'SeTimeZonePrivilege')

    def test_hLsarLookupPrivilegeValue_hLsarLookupPrivilegeName(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarLookupPrivilegeValue(dce, policyHandle, 'SeTimeZonePrivilege')
        resp.dump()
        resp = lsad.hLsarLookupPrivilegeName(dce, policyHandle, resp['Value'])
        resp.dump()
        self.assertEqual(resp['Name'], 'SeTimeZonePrivilege')

    def test_LsarLookupPrivilegeDisplayName(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarLookupPrivilegeDisplayName()
        request['PolicyHandle'] = policyHandle
        request['Name'] = 'SeTimeZonePrivilege'
        request['ClientLanguage'] = 1
        request['ClientSystemDefaultLanguage'] = 1
        resp = dce.request(request)
        resp.dump()

    def test_LsarQuerySecurityObject_LsarSetSecurityObject(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQuerySecurityObject()
        request['PolicyHandle'] = policyHandle
        request['SecurityInformation'] = lsad.OWNER_SECURITY_INFORMATION
        resp = dce.request(request)
        resp.dump()
        self.assertEqual(resp['SecurityDescriptor']['Length'], len(resp['SecurityDescriptor']['SecurityDescriptor']))
        request = lsad.LsarSetSecurityObject()
        request['PolicyHandle'] = policyHandle
        request['SecurityInformation'] = lsad.OWNER_SECURITY_INFORMATION
        request['SecurityDescriptor'] = resp['SecurityDescriptor']
        resp = dce.request(request)
        resp.dump()

    def test_hLsarQuerySecurityObject_hLsarSetSecurityObject(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarQuerySecurityObject(dce, policyHandle, lsad.OWNER_SECURITY_INFORMATION)
        hexdump(resp)
        resp = lsad.hLsarSetSecurityObject(dce, policyHandle, lsad.OWNER_SECURITY_INFORMATION, resp)
        resp.dump()

    def test_LsarQueryForestTrustInformation(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQueryForestTrustInformation()
        request['PolicyHandle'] = policyHandle
        request['TrustedDomainName'] = 'CORE'
        request['HighestRecordType'] = lsad.LSA_FOREST_TRUST_RECORD_TYPE.ForestTrustTopLevelName
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            if str(e).find('STATUS_NO_SUCH_DOMAIN') < 0:
                raise

    def test_LsarSetInformationPolicy2(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQueryInformationPolicy2()
        request['PolicyHandle'] = policyHandle
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation
        resp = dce.request(request)
        resp.dump()
        oldValue = resp['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode']
        req = lsad.LsarSetInformationPolicy2()
        req['PolicyHandle'] = policyHandle
        req['InformationClass'] = request['InformationClass']
        req['PolicyInformation'] = resp['PolicyInformation']
        req['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode'] = 0
        resp2 = dce.request(req)
        resp2.dump()
        resp = dce.request(request)
        resp.dump()
        req['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode'] = oldValue
        resp2 = dce.request(req)
        resp2.dump()

    def test_hLsarSetInformationPolicy2(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation)
        resp.dump()
        oldValue = resp['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode']
        resp['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode'] = 0
        resp2 = lsad.hLsarSetInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation, resp['PolicyInformation'])
        resp2.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation)
        resp.dump()
        resp['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode'] = oldValue
        resp2 = lsad.hLsarSetInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation, resp['PolicyInformation'])
        resp2.dump()

    def test_LsarSetInformationPolicy(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        request = lsad.LsarQueryInformationPolicy()
        request['PolicyHandle'] = policyHandle
        request['InformationClass'] = lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation
        resp = dce.request(request)
        resp.dump()
        oldValue = resp['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode']
        req = lsad.LsarSetInformationPolicy()
        req['PolicyHandle'] = policyHandle
        req['InformationClass'] = request['InformationClass']
        req['PolicyInformation'] = resp['PolicyInformation']
        req['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode'] = 0
        resp2 = dce.request(req)
        resp2.dump()
        resp = dce.request(request)
        resp.dump()
        req['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode'] = oldValue
        resp2 = dce.request(req)
        resp2.dump()

    def test_hLsarSetInformationPolicy(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        policyHandle = self.open_policy(dce)
        resp = lsad.hLsarQueryInformationPolicy(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation)
        resp.dump()
        oldValue = resp['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode']
        resp['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode'] = 0
        resp2 = lsad.hLsarSetInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation, resp['PolicyInformation'])
        resp2.dump()
        resp = lsad.hLsarQueryInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation)
        resp.dump()
        resp['PolicyInformation']['PolicyAuditEventsInfo']['AuditingMode'] = oldValue
        resp2 = lsad.hLsarSetInformationPolicy2(dce, policyHandle, lsad.POLICY_INFORMATION_CLASS.PolicyAuditEventsInformation, resp['PolicyInformation'])
        resp2.dump()

@pytest.mark.remote
class LSADTestsSMBTransport(LSADTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class LSADTestsSMBTransport64(LSADTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64
if __name__ == '__main__':
    unittest.main(verbosity=1)