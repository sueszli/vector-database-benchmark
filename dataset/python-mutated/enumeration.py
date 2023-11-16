"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.data import logger
from plugins.generic.enumeration import Enumeration as GenericEnumeration

class Enumeration(GenericEnumeration):

    def getPasswordHashes(self):
        if False:
            while True:
                i = 10
        warnMsg = 'on Virtuoso it is not possible to enumerate the user password hashes'
        logger.warning(warnMsg)
        return {}

    def getPrivileges(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        warnMsg = 'on Virtuoso it is not possible to enumerate the user privileges'
        logger.warning(warnMsg)
        return {}

    def getRoles(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        warnMsg = 'on Virtuoso it is not possible to enumerate the user roles'
        logger.warning(warnMsg)
        return {}

    def searchDb(self):
        if False:
            print('Hello World!')
        warnMsg = 'on Virtuoso it is not possible to search databases'
        logger.warning(warnMsg)
        return []

    def searchTable(self):
        if False:
            for i in range(10):
                print('nop')
        warnMsg = 'on Virtuoso it is not possible to search tables'
        logger.warning(warnMsg)
        return []

    def searchColumn(self):
        if False:
            print('Hello World!')
        warnMsg = 'on Virtuoso it is not possible to search columns'
        logger.warning(warnMsg)
        return []

    def search(self):
        if False:
            return 10
        warnMsg = 'on Virtuoso search option is not available'
        logger.warning(warnMsg)

    def getStatements(self):
        if False:
            i = 10
            return i + 15
        warnMsg = 'on Virtuoso it is not possible to enumerate the SQL statements'
        logger.warning(warnMsg)
        return []