class Solution(object):

    def pyramidTransition(self, bottom, allowed):
        if False:
            i = 10
            return i + 15
        '\n        :type bottom: str\n        :type allowed: List[str]\n        :rtype: bool\n        '

        def pyramidTransitionHelper(bottom, edges, lookup):
            if False:
                return 10

            def dfs(bottom, edges, new_bottom, idx, lookup):
                if False:
                    i = 10
                    return i + 15
                if idx == len(bottom) - 1:
                    return pyramidTransitionHelper(''.join(new_bottom), edges, lookup)
                for i in edges[ord(bottom[idx]) - ord('A')][ord(bottom[idx + 1]) - ord('A')]:
                    new_bottom[idx] = chr(i + ord('A'))
                    if dfs(bottom, edges, new_bottom, idx + 1, lookup):
                        return True
                return False
            if len(bottom) == 1:
                return True
            if bottom in lookup:
                return False
            lookup.add(bottom)
            for i in xrange(len(bottom) - 1):
                if not edges[ord(bottom[i]) - ord('A')][ord(bottom[i + 1]) - ord('A')]:
                    return False
            new_bottom = ['A'] * (len(bottom) - 1)
            return dfs(bottom, edges, new_bottom, 0, lookup)
        edges = [[[] for _ in xrange(7)] for _ in xrange(7)]
        for s in allowed:
            edges[ord(s[0]) - ord('A')][ord(s[1]) - ord('A')].append(ord(s[2]) - ord('A'))
        return pyramidTransitionHelper(bottom, edges, set())