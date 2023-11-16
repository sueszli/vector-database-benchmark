class Solution(object):

    def strongPasswordCheckerII(self, password):
        if False:
            i = 10
            return i + 15
        '\n        :type password: str\n        :rtype: bool\n        '
        SPECIAL = set('!@#$%^&*()-+')
        return len(password) >= 8 and any((c.islower() for c in password)) and any((c.isupper() for c in password)) and any((c.isdigit() for c in password)) and any((c in SPECIAL for c in password)) and all((password[i] != password[i + 1] for i in xrange(len(password) - 1)))