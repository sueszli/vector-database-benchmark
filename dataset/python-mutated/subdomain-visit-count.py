import collections

class Solution(object):

    def subdomainVisits(self, cpdomains):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type cpdomains: List[str]\n        :rtype: List[str]\n        '
        result = collections.defaultdict(int)
        for domain in cpdomains:
            (count, domain) = domain.split()
            count = int(count)
            frags = domain.split('.')
            curr = []
            for i in reversed(xrange(len(frags))):
                curr.append(frags[i])
                result['.'.join(reversed(curr))] += count
        return ['{} {}'.format(count, domain) for (domain, count) in result.iteritems()]