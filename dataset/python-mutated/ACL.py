from __future__ import absolute_import, print_function
import sys
from .BaseUtils import getTreeFromXml, encode_to_s3, decode_from_s3
from .Utils import deunicodise
try:
    import xml.etree.ElementTree as ET
except ImportError:
    import elementtree.ElementTree as ET
PY3 = sys.version_info >= (3, 0)

class Grantee(object):
    ALL_USERS_URI = 'http://acs.amazonaws.com/groups/global/AllUsers'
    LOG_DELIVERY_URI = 'http://acs.amazonaws.com/groups/s3/LogDelivery'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.xsi_type = None
        self.tag = None
        self.name = None
        self.display_name = ''
        self.permission = None

    def __repr__(self):
        if False:
            while True:
                i = 10
        return repr('Grantee("%(tag)s", "%(name)s", "%(permission)s")' % {'tag': self.tag, 'name': self.name, 'permission': self.permission})

    def isAllUsers(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tag == 'URI' and self.name == Grantee.ALL_USERS_URI

    def isAnonRead(self):
        if False:
            i = 10
            return i + 15
        return self.isAllUsers() and (self.permission == 'READ' or self.permission == 'FULL_CONTROL')

    def isAnonWrite(self):
        if False:
            while True:
                i = 10
        return self.isAllUsers() and (self.permission == 'WRITE' or self.permission == 'FULL_CONTROL')

    def getElement(self):
        if False:
            return 10
        el = ET.Element('Grant')
        grantee = ET.SubElement(el, 'Grantee', {'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance', 'xsi:type': self.xsi_type})
        name = ET.SubElement(grantee, self.tag)
        name.text = self.name
        permission = ET.SubElement(el, 'Permission')
        permission.text = self.permission
        return el

class GranteeAnonRead(Grantee):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        Grantee.__init__(self)
        self.xsi_type = 'Group'
        self.tag = 'URI'
        self.name = Grantee.ALL_USERS_URI
        self.permission = 'READ'

class GranteeLogDelivery(Grantee):

    def __init__(self, permission):
        if False:
            i = 10
            return i + 15
        '\n        permission must be either READ_ACP or WRITE\n        '
        Grantee.__init__(self)
        self.xsi_type = 'Group'
        self.tag = 'URI'
        self.name = Grantee.LOG_DELIVERY_URI
        self.permission = permission

class ACL(object):
    EMPTY_ACL = b'<AccessControlPolicy><Owner><ID></ID></Owner><AccessControlList></AccessControlList></AccessControlPolicy>'

    def __init__(self, xml=None):
        if False:
            i = 10
            return i + 15
        if not xml:
            xml = ACL.EMPTY_ACL
        self.grantees = []
        self.owner_id = ''
        self.owner_nick = ''
        tree = getTreeFromXml(encode_to_s3(xml))
        self.parseOwner(tree)
        self.parseGrants(tree)

    def parseOwner(self, tree):
        if False:
            for i in range(10):
                print('nop')
        self.owner_id = tree.findtext('.//Owner//ID')
        self.owner_nick = tree.findtext('.//Owner//DisplayName')

    def parseGrants(self, tree):
        if False:
            i = 10
            return i + 15
        for grant in tree.findall('.//Grant'):
            grantee = Grantee()
            g = grant.find('.//Grantee')
            grantee.xsi_type = g.attrib['{http://www.w3.org/2001/XMLSchema-instance}type']
            grantee.permission = grant.find('Permission').text
            for el in g:
                if el.tag == 'DisplayName':
                    grantee.display_name = el.text
                else:
                    grantee.tag = el.tag
                    grantee.name = el.text
            self.grantees.append(grantee)

    def getGrantList(self):
        if False:
            return 10
        acl = []
        for grantee in self.grantees:
            if grantee.display_name:
                user = grantee.display_name
            elif grantee.isAllUsers():
                user = '*anon*'
            else:
                user = grantee.name
            acl.append({'grantee': user, 'permission': grantee.permission})
        return acl

    def getOwner(self):
        if False:
            return 10
        return {'id': self.owner_id, 'nick': self.owner_nick}

    def isAnonRead(self):
        if False:
            while True:
                i = 10
        for grantee in self.grantees:
            if grantee.isAnonRead():
                return True
        return False

    def isAnonWrite(self):
        if False:
            i = 10
            return i + 15
        for grantee in self.grantees:
            if grantee.isAnonWrite():
                return True
        return False

    def grantAnonRead(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.isAnonRead():
            self.appendGrantee(GranteeAnonRead())

    def revokeAnonRead(self):
        if False:
            while True:
                i = 10
        self.grantees = [g for g in self.grantees if not g.isAnonRead()]

    def revokeAnonWrite(self):
        if False:
            print('Hello World!')
        self.grantees = [g for g in self.grantees if not g.isAnonWrite()]

    def appendGrantee(self, grantee):
        if False:
            print('Hello World!')
        self.grantees.append(grantee)

    def hasGrant(self, name, permission):
        if False:
            while True:
                i = 10
        name = name.lower()
        permission = permission.upper()
        for grantee in self.grantees:
            if grantee.name.lower() == name:
                if grantee.permission == 'FULL_CONTROL':
                    return True
                elif grantee.permission.upper() == permission:
                    return True
        return False

    def grant(self, name, permission):
        if False:
            print('Hello World!')
        if self.hasGrant(name, permission):
            return
        permission = permission.upper()
        if 'ALL' == permission:
            permission = 'FULL_CONTROL'
        if 'FULL_CONTROL' == permission:
            self.revoke(name, 'ALL')
        grantee = Grantee()
        grantee.name = name
        grantee.permission = permission
        if '@' in name:
            grantee.name = grantee.name.lower()
            grantee.xsi_type = 'AmazonCustomerByEmail'
            grantee.tag = 'EmailAddress'
        elif 'http://acs.amazonaws.com/groups/' in name:
            grantee.xsi_type = 'Group'
            grantee.tag = 'URI'
        else:
            grantee.name = grantee.name.lower()
            grantee.xsi_type = 'CanonicalUser'
            grantee.tag = 'ID'
        self.appendGrantee(grantee)

    def revoke(self, name, permission):
        if False:
            i = 10
            return i + 15
        name = name.lower()
        permission = permission.upper()
        if 'ALL' == permission:
            self.grantees = [g for g in self.grantees if not (g.name.lower() == name or (g.display_name is not None and g.display_name.lower() == name))]
        else:
            self.grantees = [g for g in self.grantees if not ((g.display_name is not None and g.display_name.lower() == name or g.name.lower() == name) and g.permission.upper() == permission)]

    def get_printable_tree(self):
        if False:
            while True:
                i = 10
        tree = getTreeFromXml(ACL.EMPTY_ACL)
        tree.attrib['xmlns'] = 'http://s3.amazonaws.com/doc/2006-03-01/'
        owner = tree.find('.//Owner//ID')
        owner.text = self.owner_id
        acl = tree.find('.//AccessControlList')
        for grantee in self.grantees:
            acl.append(grantee.getElement())
        return tree

    def __unicode__(self):
        if False:
            i = 10
            return i + 15
        return decode_from_s3(ET.tostring(self.get_printable_tree()))

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if PY3:
            return ET.tostring(self.get_printable_tree(), encoding='unicode')
        else:
            return ET.tostring(self.get_printable_tree())
if __name__ == '__main__':
    xml = b'<?xml version="1.0" encoding="UTF-8"?>\n<AccessControlPolicy xmlns="http://s3.amazonaws.com/doc/2006-03-01/">\n<Owner>\n    <ID>12345678901234567890</ID>\n    <DisplayName>owner-nickname</DisplayName>\n</Owner>\n<AccessControlList>\n    <Grant>\n        <Grantee xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="CanonicalUser">\n            <ID>12345678901234567890</ID>\n            <DisplayName>owner-nickname</DisplayName>\n        </Grantee>\n        <Permission>FULL_CONTROL</Permission>\n    </Grant>\n    <Grant>\n        <Grantee xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="Group">\n            <URI>http://acs.amazonaws.com/groups/global/AllUsers</URI>\n        </Grantee>\n        <Permission>READ</Permission>\n    </Grant>\n</AccessControlList>\n</AccessControlPolicy>\n    '
    acl = ACL(xml)
    print('Grants:', acl.getGrantList())
    acl.revokeAnonRead()
    print('Grants:', acl.getGrantList())
    acl.grantAnonRead()
    print('Grants:', acl.getGrantList())
    print(acl)