class TreeAncestor(object):

    def __init__(self, n, parent):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type parent: List[int]\n        '
        par = [[p] if p != -1 else [] for p in parent]
        q = [par[i] for (i, p) in enumerate(parent) if p != -1]
        i = 0
        while q:
            new_q = []
            for p in q:
                if not i < len(par[p[i]]):
                    continue
                p.append(par[p[i]][i])
                new_q.append(p)
            q = new_q
            i += 1
        self.__parent = par

    def getKthAncestor(self, node, k):
        if False:
            i = 10
            return i + 15
        '\n        :type node: int\n        :type k: int\n        :rtype: int\n        '
        (par, i, pow_i_of_2) = (self.__parent, 0, 1)
        while pow_i_of_2 <= k:
            if k & pow_i_of_2:
                if not i < len(par[node]):
                    return -1
                node = par[node][i]
            i += 1
            pow_i_of_2 *= 2
        return node