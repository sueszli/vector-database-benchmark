import socket
import os
import struct
import time
import select
ICMP_ECHO_REQUEST = 8

def checksum(str):
    if False:
        for i in range(10):
            print('nop')
    csum = 0
    countTo = len(str) / 2 * 2
    count = 0
    while count < countTo:
        thisVal = str[count + 1] * 256 + str[count]
        csum = csum + thisVal
        csum = csum & 4294967295
        count = count + 2
    if countTo < len(str):
        csum = csum + str[len(str) - 1].decode()
        csum = csum & 4294967295
    csum = (csum >> 16) + (csum & 65535)
    csum = csum + (csum >> 16)
    answer = ~csum
    answer = answer & 65535
    answer = answer >> 8 | answer << 8 & 65280
    return answer

def receiveOnePing(mySocket, ID, sequence, destAddr, timeout):
    if False:
        while True:
            i = 10
    timeLeft = timeout
    while 1:
        startedSelect = time.time()
        whatReady = select.select([mySocket], [], [], timeLeft)
        howLongInSelect = time.time() - startedSelect
        if whatReady[0] == []:
            return None
        timeReceived = time.time()
        (recPacket, addr) = mySocket.recvfrom(1024)
        header = recPacket[20:28]
        (type, code, checksum, packetID, sequence) = struct.unpack('!bbHHh', header)
        if type == 0 and packetID == ID:
            byte_in_double = struct.calcsize('!d')
            timeSent = struct.unpack('!d', recPacket[28:28 + byte_in_double])[0]
            delay = timeReceived - timeSent
            ttl = ord(struct.unpack('!c', recPacket[8:9])[0].decode())
            return (delay, ttl, byte_in_double)
        timeLeft = timeLeft - howLongInSelect
        if timeLeft <= 0:
            return None

def sendOnePing(mySocket, ID, sequence, destAddr):
    if False:
        while True:
            i = 10
    myChecksum = 0
    header = struct.pack('!bbHHh', ICMP_ECHO_REQUEST, 0, myChecksum, ID, sequence)
    data = struct.pack('!d', time.time())
    myChecksum = checksum(header + data)
    header = struct.pack('!bbHHh', ICMP_ECHO_REQUEST, 0, myChecksum, ID, sequence)
    packet = header + data
    mySocket.sendto(packet, (destAddr, 1))

def doOnePing(destAddr, ID, sequence, timeout):
    if False:
        print('Hello World!')
    icmp = socket.getprotobyname('icmp')
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_RAW, icmp)
    sendOnePing(mySocket, ID, sequence, destAddr)
    delay = receiveOnePing(mySocket, ID, sequence, destAddr, timeout)
    mySocket.close()
    return delay

def ping(host, timeout=1):
    if False:
        while True:
            i = 10
    dest = socket.gethostbyname(host)
    print('Pinging ' + dest + ' using Python:')
    print('')
    myID = os.getpid() & 65535
    loss = 0
    for i in range(4):
        result = doOnePing(dest, myID, i, timeout)
        if not result:
            print('Request timed out.')
            loss += 1
        else:
            delay = int(result[0] * 1000)
            ttl = result[1]
            bytes = result[2]
            print('Received from ' + dest + ': byte(s)=' + str(bytes) + ' delay=' + str(delay) + 'ms TTL=' + str(ttl))
        time.sleep(1)
    print('Packet: sent = ' + str(4) + ' received = ' + str(4 - loss) + ' lost = ' + str(loss))
    return
ping('www.baidu.com')