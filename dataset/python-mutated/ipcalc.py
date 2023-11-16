import argparse
from ivre import utils

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Tool for ip addresses manipulation.')
    parser.add_argument('ips', nargs='*', help='Display results for specified IP addresses or ranges.')
    args = parser.parse_args()
    while '-' in args.ips:
        idx = args.ips.index('-')
        args.ips = args.ips[:idx - 1] + ['%s-%s' % (args.ips[idx - 1], args.ips[idx + 1])] + args.ips[idx + 2:]
    for a in args.ips:
        if '/' in a:
            a = utils.net2range(a)
            print('%s-%s' % (a[0], a[1]))
        elif '-' in a:
            a = a.split('-', 1)
            if a[0].isdigit():
                a[0] = int(a[0])
            if a[1].isdigit():
                a[1] = int(a[1])
            for n in utils.range2nets((a[0], a[1])):
                print(n)
        else:
            if a.isdigit():
                a = utils.force_int2ip(int(a))
            else:
                a = utils.force_ip2int(a)
            print(a)