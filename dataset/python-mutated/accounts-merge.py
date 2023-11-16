import collections

class UnionFind(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.set = []

    def get_id(self):
        if False:
            while True:
                i = 10
        self.set.append(len(self.set))
        return len(self.set) - 1

    def find_set(self, x):
        if False:
            print('Hello World!')
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            i = 10
            return i + 15
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root != y_root:
            self.set[min(x_root, y_root)] = max(x_root, y_root)

class Solution(object):

    def accountsMerge(self, accounts):
        if False:
            print('Hello World!')
        '\n        :type accounts: List[List[str]]\n        :rtype: List[List[str]]\n        '
        union_find = UnionFind()
        email_to_name = {}
        email_to_id = {}
        for account in accounts:
            name = account[0]
            for i in xrange(1, len(account)):
                if account[i] not in email_to_id:
                    email_to_name[account[i]] = name
                    email_to_id[account[i]] = union_find.get_id()
                union_find.union_set(email_to_id[account[1]], email_to_id[account[i]])
        result = collections.defaultdict(list)
        for email in email_to_name.keys():
            result[union_find.find_set(email_to_id[email])].append(email)
        for emails in result.values():
            emails.sort()
        return [[email_to_name[emails[0]]] + emails for emails in result.values()]