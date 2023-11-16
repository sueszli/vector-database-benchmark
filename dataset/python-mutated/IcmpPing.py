from socket import *
import os
import sys
import struct
import time
import select
import binascii
ICMP_ECHO_REQUEST = 8

def checksum(string):
    if False:
        return 10
    csum = 0
    countTo = len(string) // 2 * 2
    count = 0
    while count < countTo:
        thisVal = ord(string[count + 1]) * 256 + ord(string[count])
        csum = csum + thisVal
        csum = csum & 4294967295
        count = count + 2
    if countTo < len(string):
        csum = csum + ord(string[len(string) - 1])
        csum = csum & 4294967295
    csum = (csum >> 16) + (csum & 65535)
    csum = csum + (csum >> 16)
    answer = ~csum
    answer = answer & 65535
    answer = answer >> 8 | answer << 8 & 65280
    return answer

def receiveOnePing(mySocket, ID, timeout, destAddr):
    if False:
        while True:
            i = 10
    timeLeft = timeout
    while 1:
        startedSelect = time.time()
        whatReady = select.select([mySocket], [], [], timeLeft)
        howLongInSelect = time.time() - startedSelect
        if whatReady[0] == []:
            return 'Request timed out.'
        timeReceived = time.time()
        (recPacket, addr) = mySocket.recvfrom(1024)
        icmpHeader = recPacket[20:28]
        rawTTL = struct.unpack('s', bytes([recPacket[8]]))[0]
        TTL = int(binascii.hexlify(rawTTL), 16)
        (icmpType, code, checksum, packetID, sequence) = struct.unpack('bbHHh', icmpHeader)
        if packetID == ID:
            byte = struct.calcsize('d')
            timeSent = struct.unpack('d', recPacket[28:28 + byte])[0]
            return 'Reply from %s: bytes=%d time=%f5ms TTL=%d' % (destAddr, len(recPacket), (timeReceived - timeSent) * 1000, TTL)
        timeLeft = timeLeft - howLongInSelect
        if timeLeft <= 0:
            return 'Request timed out.'

def sendOnePing(mySocket, destAddr, ID):
    if False:
        for i in range(10):
            print('nop')
    myChecksum = 0
    header = struct.pack('bbHHh', ICMP_ECHO_REQUEST, 0, myChecksum, ID, 1)
    data = struct.pack('d', time.time())
    myChecksum = checksum(str(header + data))
    if sys.platform == 'darwin':
        myChecksum = htons(myChecksum) & 65535
    else:
        myChecksum = htons(myChecksum)
    header = struct.pack('bbHHh', ICMP_ECHO_REQUEST, 0, myChecksum, ID, 1)
    packet = header + data
    mySocket.sendto(packet, (destAddr, 1))

def doOnePing(destAddr, timeout):
    if False:
        while True:
            i = 10
    icmp = getprotobyname('icmp')
    mySocket = socket(AF_INET, SOCK_RAW, icmp)
    myID = os.getpid() & 65535
    sendOnePing(mySocket, destAddr, myID)
    delay = receiveOnePing(mySocket, myID, timeout, destAddr)
    mySocket.close()
    return delay

def ping(host, timeout=1):
    if False:
        i = 10
        return i + 15
    dest = gethostbyname(host)
    print('Pinging ' + dest + ' using Python:')
    print('')
    while True:
        delay = doOnePing(dest, timeout)
        print(delay)
        time.sleep(1)
    return delay
ping('google.com')