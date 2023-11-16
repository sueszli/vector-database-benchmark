import sys
import pywintypes
from ntsecuritycon import *
from win32net import NetUserModalsGet
from win32security import LookupAccountSid

def LookupAliasFromRid(TargetComputer, Rid):
    if False:
        print('Hello World!')
    sid = pywintypes.SID()
    sid.Initialize(SECURITY_NT_AUTHORITY, 2)
    for (i, r) in enumerate((SECURITY_BUILTIN_DOMAIN_RID, Rid)):
        sid.SetSubAuthority(i, r)
    (name, domain, typ) = LookupAccountSid(TargetComputer, sid)
    return name

def LookupUserGroupFromRid(TargetComputer, Rid):
    if False:
        return 10
    umi2 = NetUserModalsGet(TargetComputer, 2)
    domain_sid = umi2['domain_id']
    SubAuthorityCount = domain_sid.GetSubAuthorityCount()
    sid = pywintypes.SID()
    sid.Initialize(domain_sid.GetSidIdentifierAuthority(), SubAuthorityCount + 1)
    for i in range(SubAuthorityCount):
        sid.SetSubAuthority(i, domain_sid.GetSubAuthority(i))
    sid.SetSubAuthority(SubAuthorityCount, Rid)
    (name, domain, typ) = LookupAccountSid(TargetComputer, sid)
    return name

def main():
    if False:
        for i in range(10):
            print('nop')
    if len(sys.argv) == 2:
        targetComputer = sys.argv[1]
    else:
        targetComputer = None
    name = LookupUserGroupFromRid(targetComputer, DOMAIN_USER_RID_ADMIN)
    print(f"'Administrator' user name = {name}")
    name = LookupAliasFromRid(targetComputer, DOMAIN_ALIAS_RID_ADMINS)
    print(f"'Administrators' local group/alias name = {name}")
if __name__ == '__main__':
    main()