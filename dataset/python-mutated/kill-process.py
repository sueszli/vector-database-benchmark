import collections

class Solution(object):

    def killProcess(self, pid, ppid, kill):
        if False:
            return 10
        '\n        :type pid: List[int]\n        :type ppid: List[int]\n        :type kill: int\n        :rtype: List[int]\n        '

        def killAll(pid, children, killed):
            if False:
                return 10
            killed.append(pid)
            for child in children[pid]:
                killAll(child, children, killed)
        result = []
        children = collections.defaultdict(set)
        for i in xrange(len(pid)):
            children[ppid[i]].add(pid[i])
        killAll(kill, children, result)
        return result

class Solution2(object):

    def killProcess(self, pid, ppid, kill):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type pid: List[int]\n        :type ppid: List[int]\n        :type kill: int\n        :rtype: List[int]\n        '

        def killAll(pid, children, killed):
            if False:
                i = 10
                return i + 15
            killed.append(pid)
            for child in children[pid]:
                killAll(child, children, killed)
        result = []
        children = collections.defaultdict(set)
        for i in xrange(len(pid)):
            children[ppid[i]].add(pid[i])
        q = collections.deque()
        q.append(kill)
        while q:
            p = q.popleft()
            result.append(p)
            for child in children[p]:
                q.append(child)
        return result