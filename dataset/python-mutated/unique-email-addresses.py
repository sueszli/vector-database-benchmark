class Solution(object):

    def numUniqueEmails(self, emails):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type emails: List[str]\n        :rtype: int\n        '

        def convert(email):
            if False:
                return 10
            (name, domain) = email.split('@')
            name = name[:name.index('+')]
            return ''.join([''.join(name.split('.')), '@', domain])
        lookup = set()
        for email in emails:
            lookup.add(convert(email))
        return len(lookup)