import collections

class Solution(object):

    def sequenceReconstruction(self, org, seqs):
        if False:
            i = 10
            return i + 15
        '\n        :type org: List[int]\n        :type seqs: List[List[int]]\n        :rtype: bool\n        '
        if not seqs:
            return False
        pos = [0] * (len(org) + 1)
        for i in xrange(len(org)):
            pos[org[i]] = i
        is_matched = [False] * (len(org) + 1)
        cnt_to_match = len(org) - 1
        for seq in seqs:
            for i in xrange(len(seq)):
                if not 0 < seq[i] <= len(org):
                    return False
                if i == 0:
                    continue
                if pos[seq[i - 1]] >= pos[seq[i]]:
                    return False
                if is_matched[seq[i - 1]] == False and pos[seq[i - 1]] + 1 == pos[seq[i]]:
                    is_matched[seq[i - 1]] = True
                    cnt_to_match -= 1
        return cnt_to_match == 0

class Solution2(object):

    def sequenceReconstruction(self, org, seqs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type org: List[int]\n        :type seqs: List[List[int]]\n        :rtype: bool\n        '
        graph = collections.defaultdict(set)
        indegree = collections.defaultdict(int)
        integer_set = set()
        for seq in seqs:
            for i in seq:
                integer_set.add(i)
            if len(seq) == 1:
                if seq[0] not in indegree:
                    indegree[seq[0]] = 0
                continue
            for i in xrange(len(seq) - 1):
                if seq[i] not in indegree:
                    indegree[seq[i]] = 0
                if seq[i + 1] not in graph[seq[i]]:
                    graph[seq[i]].add(seq[i + 1])
                    indegree[seq[i + 1]] += 1
        cnt_of_zero_indegree = 0
        res = []
        q = []
        for i in indegree:
            if indegree[i] == 0:
                cnt_of_zero_indegree += 1
                if cnt_of_zero_indegree > 1:
                    return False
                q.append(i)
        while q:
            i = q.pop()
            res.append(i)
            cnt_of_zero_indegree = 0
            for j in graph[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    cnt_of_zero_indegree += 1
                    if cnt_of_zero_indegree > 1:
                        return False
                    q.append(j)
        return res == org and len(org) == len(integer_set)