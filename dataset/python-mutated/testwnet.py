import os
import win32api
import win32wnet
from winnetwk import *
possible_shares = []

def _doDumpHandle(handle, level=0):
    if False:
        i = 10
        return i + 15
    indent = ' ' * level
    while 1:
        items = win32wnet.WNetEnumResource(handle, 0)
        if len(items) == 0:
            break
        for item in items:
            try:
                if item.dwDisplayType == RESOURCEDISPLAYTYPE_SHARE:
                    print(indent + 'Have share with name:', item.lpRemoteName)
                    possible_shares.append(item)
                elif item.dwDisplayType == RESOURCEDISPLAYTYPE_GENERIC:
                    print(indent + 'Have generic resource with name:', item.lpRemoteName)
                else:
                    print(indent + 'Enumerating ' + item.lpRemoteName, end=' ')
                    k = win32wnet.WNetOpenEnum(RESOURCE_GLOBALNET, RESOURCETYPE_ANY, 0, item)
                    print()
                    _doDumpHandle(k, level + 1)
                    win32wnet.WNetCloseEnum(k)
            except win32wnet.error as details:
                print(indent + "Couldn't enumerate this resource: " + details.strerror)

def TestOpenEnum():
    if False:
        for i in range(10):
            print('nop')
    print('Enumerating all resources on the network - this may take some time...')
    handle = win32wnet.WNetOpenEnum(RESOURCE_GLOBALNET, RESOURCETYPE_ANY, 0, None)
    try:
        _doDumpHandle(handle)
    finally:
        handle.Close()
    print('Finished dumping all resources.')

def findUnusedDriveLetter():
    if False:
        while True:
            i = 10
    existing = [x[0].lower() for x in win32api.GetLogicalDriveStrings().split('\x00') if x]
    handle = win32wnet.WNetOpenEnum(RESOURCE_REMEMBERED, RESOURCETYPE_DISK, 0, None)
    try:
        while 1:
            items = win32wnet.WNetEnumResource(handle, 0)
            if len(items) == 0:
                break
            xtra = [i.lpLocalName[0].lower() for i in items if i.lpLocalName]
            existing.extend(xtra)
    finally:
        handle.Close()
    for maybe in 'defghijklmnopqrstuvwxyz':
        if maybe not in existing:
            return maybe
    raise RuntimeError('All drive mappings are taken?')

def TestConnection():
    if False:
        print('Hello World!')
    if len(possible_shares) == 0:
        print("Couldn't find any potential shares to connect to")
        return
    localName = findUnusedDriveLetter() + ':'
    for share in possible_shares:
        print('Attempting connection of', localName, 'to', share.lpRemoteName)
        try:
            win32wnet.WNetAddConnection2(share.dwType, localName, share.lpRemoteName)
        except win32wnet.error as details:
            print("Couldn't connect: " + details.strerror)
            continue
        try:
            fname = os.path.join(localName + '\\', os.listdir(localName + '\\')[0])
            try:
                print("Universal name of '{}' is '{}'".format(fname, win32wnet.WNetGetUniversalName(fname)))
            except win32wnet.error as details:
                print(f"Couldn't get universal name of '{fname}': {details.strerror}")
            print('User name for this connection is', win32wnet.WNetGetUser(localName))
        finally:
            win32wnet.WNetCancelConnection2(localName, 0, 0)
        nr = win32wnet.NETRESOURCE()
        nr.dwType = share.dwType
        nr.lpLocalName = localName
        nr.lpRemoteName = share.lpRemoteName
        win32wnet.WNetAddConnection2(nr)
        win32wnet.WNetCancelConnection2(localName, 0, 0)
        win32wnet.WNetAddConnection3(0, nr)
        win32wnet.WNetCancelConnection2(localName, 0, 0)
        break

def TestGetUser():
    if False:
        return 10
    u = win32wnet.WNetGetUser()
    print('Current global user is', repr(u))
    if u != win32wnet.WNetGetUser(None):
        raise RuntimeError('Default value didnt seem to work!')
TestGetUser()
TestOpenEnum()
TestConnection()