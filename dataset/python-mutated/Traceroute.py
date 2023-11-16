from socket import *
import os
import sys
import struct
import time
import select
import binascii
ICMP_ECHO_REQUEST = 8
MAX_HOPS = 30
TIMEOUT = 2.0
TRIES = 2

def checksum(string):
    if False:
        while True:
            i = 10
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

def build_packet():
    if False:
        return 10
    myChecksum = 0
    header = struct.pack('bbHh', ICMP_ECHO_REQUEST, 0, myChecksum, 1)
    data = struct.pack('d', time.time())
    myChecksum = checksum(str(header + data))
    if sys.platform == 'darwin':
        myChecksum = htons(myChecksum) & 65535
    else:
        myChecksum = htons(myChecksum)
    header = struct.pack('bbHHh', ICMP_ECHO_REQUEST, 0, myChecksum, os.getpid() & 65535, 1)
    packet = header + data
    return packet

def get_route(hostname):
    if False:
        i = 10
        return i + 15
    timeLeft = TIMEOUT
    for ttl in range(1, MAX_HOPS):
        for tries in range(TRIES):
            destAddr = gethostbyname(hostname)
            mySocket = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)
            mySocket.settimeout(TIMEOUT)
            mySocket.bind(('', 0))
            mySocket.setsockopt(IPPROTO_IP, IP_TTL, struct.pack('I', ttl))
            mySocket.settimeout(TIMEOUT)
            try:
                d = build_packet()
                mySocket.sendto(d, (hostname, 0))
                t = time.time()
                startedSelect = time.time()
                whatReady = select.select([mySocket], [], [], timeLeft)
                howLongInSelect = time.time() - startedSelect
                if whatReady[0] == []:
                    print('  *        *        *    Request timed out.')
                (recvPacket, addr) = mySocket.recvfrom(1024)
                timeReceived = time.time()
                timeLeft = timeLeft - howLongInSelect
                if timeLeft <= 0:
                    print('  *        *        *    Request timed out.')
            except timeout:
                continue
            else:
                (types, code) = recvPacket[20:22]
                if types == 11:
                    bytes = struct.calcsize('d')
                    timeSent = struct.unpack('d', recvPacket[28:28 + bytes])[0]
                    print('  %d    rtt=%.0f ms    %s' % (ttl, (timeReceived - t) * 1000, addr[0]))
                elif types == 3:
                    bytes = struct.calcsize('d')
                    timeSent = struct.unpack('d', recvPacket[28:28 + bytes])[0]
                    print('  %d    rtt=%.0f ms    %s' % (ttl, (timeReceived - t) * 1000, addr[0]))
                elif types == 0:
                    bytes = struct.calcsize('d')
                    timeSent = struct.unpack('d', recvPacket[28:28 + bytes])[0]
                    print('  %d    rtt=%.0f ms    %s' % (ttl, (timeReceived - timeSent) * 1000, addr[0]))
                    return
                else:
                    print('error')
                break
            finally:
                mySocket.close()
get_route('google.com')