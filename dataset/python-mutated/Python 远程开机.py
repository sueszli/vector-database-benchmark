def wake_up(request, mac='DC-4A-3E-78-3E-0A'):
    if False:
        for i in range(10):
            print('nop')
    MAC = mac
    BROADCAST = '192.168.0.255'
    if len(MAC) != 17:
        raise ValueError("MAC address should be set as form 'XX-XX-XX-XX-XX-XX'")
    mac_address = MAC.replace('-', '')
    data = ''.join(['FFFFFFFFFFFF', mac_address * 20])
    send_data = b''
    for i in range(0, len(data), 2):
        send_data = b''.join([send_data, struct.pack('B', int(data[i:i + 2], 16))])
    print(send_data)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(send_data, (BROADCAST, 7))
        time.sleep(1)
        sock.sendto(send_data, (BROADCAST, 7))
        time.sleep(1)
        sock.sendto(send_data, (BROADCAST, 7))
        return HttpResponse()
        print('Done')
    except Exception as e:
        return HttpResponse()
        print(e)