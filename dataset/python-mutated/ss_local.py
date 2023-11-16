from ssshare.shadowsocks import shell, daemon, eventloop, tcprelay, udprelay, asyncdns

def main(dictionary=None, str_json=None, port=None):
    if False:
        i = 10
        return i + 15
    shell.check_python()
    if str_json:
        config = shell.check_and_parse_config(shell.parse_json_in_str(shell.remove_comment(str_json)))
    elif dictionary:
        config = shell.check_and_parse_config(dictionary)
    else:
        raise Exception('No config specified')
    if port:
        config['local_port'] = int(port)
    if not config.get('dns_ipv6', False):
        asyncdns.IPV6_CONNECTION_SUPPORT = False
    daemon.daemon_exec(config)
    try:
        dns_resolver = asyncdns.DNSResolver()
        tcp_server = tcprelay.TCPRelay(config, dns_resolver, True)
        udp_server = udprelay.UDPRelay(config, dns_resolver, True)
        loop = eventloop.EventLoop()
        dns_resolver.add_to_loop(loop)
        tcp_server.add_to_loop(loop)
        udp_server.add_to_loop(loop)
        return [loop, tcp_server, udp_server]
        loop.run()
    except OSError as e:
        print(e)
        raise OSError(e)
    except Exception as e:
        if 'tcp_server' in locals():
            tcp_server.close(next_tick=True)
        if 'udp_server' in locals():
            udp_server.close(next_tick=True)
        if 'loop' in locals():
            loop.stop()
        shell.print_exception(e)
        raise Exception(e)
if __name__ == '__main__':
    pass