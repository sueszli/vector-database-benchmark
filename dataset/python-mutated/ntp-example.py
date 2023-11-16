import datetime
import struct
import trio

def make_query_packet():
    if False:
        print('Hello World!')
    'Construct a UDP packet suitable for querying an NTP server to ask for\n    the current time.'
    packet = bytearray(48)
    packet[0] = 227
    return packet

def extract_transmit_timestamp(ntp_packet):
    if False:
        for i in range(10):
            print('nop')
    'Given an NTP packet, extract the "transmit timestamp" field, as a\n    Python datetime.'
    encoded_transmit_timestamp = ntp_packet[40:48]
    (seconds, fraction) = struct.unpack('!II', encoded_transmit_timestamp)
    base_time = datetime.datetime(1900, 1, 1)
    offset = datetime.timedelta(seconds=seconds + fraction / 2 ** 32)
    return base_time + offset

async def main():
    print('Our clock currently reads (in UTC):', datetime.datetime.utcnow())
    servers = await trio.socket.getaddrinfo('pool.ntp.org', 'ntp', family=trio.socket.AF_INET, type=trio.socket.SOCK_DGRAM)
    query_packet = make_query_packet()
    udp_sock = trio.socket.socket(family=trio.socket.AF_INET, type=trio.socket.SOCK_DGRAM)
    print('-- Sending queries --')
    for server in servers:
        address = server[-1]
        print('Sending to:', address)
        await udp_sock.sendto(query_packet, address)
    print('-- Reading responses (for 10 seconds) --')
    with trio.move_on_after(10):
        while True:
            (data, address) = await udp_sock.recvfrom(1024)
            print('Got response from:', address)
            transmit_timestamp = extract_transmit_timestamp(data)
            print('Their clock read (in UTC):', transmit_timestamp)
trio.run(main)